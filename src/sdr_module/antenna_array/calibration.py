"""
Array calibration for antenna arrays.

Provides automated calibration of phase and amplitude offsets
between antenna array elements using known reference signals.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .array_config import ArrayConfig, ElementCalibration
from .cross_correlator import CrossCorrelator

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Calibration methods."""

    MUTUAL_COUPLING = "mutual_coupling"  # Use mutual coupling between elements
    KNOWN_SOURCE = "known_source"  # Use known source at known position
    CORRELATION = "correlation"  # Cross-correlation based
    PILOT_TONE = "pilot_tone"  # Inject pilot tone


class CalibrationState(Enum):
    """Calibration state."""

    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CalibrationMeasurement:
    """A single calibration measurement."""

    element_index: int
    reference_element: int
    phase_offset: float  # radians
    amplitude_ratio: float  # linear
    delay_samples: float
    timestamp: float
    confidence: float
    source_azimuth: Optional[float] = None
    source_elevation: Optional[float] = None


@dataclass
class CalibrationResult:
    """Result of array calibration."""

    success: bool
    method: CalibrationMethod
    reference_element: int
    measurements: List[CalibrationMeasurement]
    element_calibrations: Dict[int, ElementCalibration]
    overall_confidence: float
    timestamp: float
    duration: float
    error_message: Optional[str] = None

    def get_phase_corrections(self) -> Dict[int, float]:
        """Get phase corrections for all elements."""
        return {
            idx: cal.phase_offset for idx, cal in self.element_calibrations.items()
        }

    def get_amplitude_corrections(self) -> Dict[int, float]:
        """Get amplitude corrections for all elements."""
        return {
            idx: cal.amplitude_scale for idx, cal in self.element_calibrations.items()
        }


@dataclass
class CalibrationConfig:
    """Configuration for calibration procedure."""

    method: CalibrationMethod = CalibrationMethod.CORRELATION
    reference_element: int = 0
    num_averages: int = 10  # Number of measurements to average
    settling_time: float = 0.1  # Time to wait between measurements (seconds)
    min_confidence: float = 0.7  # Minimum confidence for valid calibration
    max_phase_change: float = np.pi  # Maximum allowed phase change (radians)
    calibration_frequency: float = 0.0  # Frequency for calibration (0 = use common)

    # Known source calibration
    source_azimuth: float = 0.0  # Known source azimuth (radians)
    source_elevation: float = 0.0  # Known source elevation (radians)

    # Pilot tone calibration
    pilot_frequency_offset: float = 100e3  # Pilot tone offset from center (Hz)
    pilot_amplitude: float = 0.1  # Pilot tone amplitude


class ArrayCalibrator:
    """
    Automated array calibration system.

    Provides methods for calibrating phase and amplitude offsets
    between antenna array elements.

    Calibration Methods:
        - CORRELATION: Use cross-correlation with ambient signals
        - KNOWN_SOURCE: Use a known source at a known position
        - PILOT_TONE: Inject a known pilot tone
        - MUTUAL_COUPLING: Use element mutual coupling

    Example:
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        # Calibrate using correlation method
        result = calibrator.calibrate_correlation(signals)

        if result.success:
            # Apply calibration to config
            calibrator.apply_calibration(result)
    """

    def __init__(
        self,
        config: ArrayConfig,
        cal_config: Optional[CalibrationConfig] = None,
    ) -> None:
        """
        Initialize array calibrator.

        Args:
            config: Array configuration
            cal_config: Calibration configuration
        """
        self._config = config
        self._cal_config = cal_config or CalibrationConfig()
        self._correlator = CrossCorrelator(sample_rate=config.common_sample_rate)

        self._state = CalibrationState.IDLE
        self._last_result: Optional[CalibrationResult] = None
        self._measurement_history: List[CalibrationResult] = []

    @property
    def config(self) -> ArrayConfig:
        """Get array configuration."""
        return self._config

    @property
    def calibration_config(self) -> CalibrationConfig:
        """Get calibration configuration."""
        return self._cal_config

    @property
    def state(self) -> CalibrationState:
        """Get current calibration state."""
        return self._state

    @property
    def last_result(self) -> Optional[CalibrationResult]:
        """Get last calibration result."""
        return self._last_result

    def calibrate_correlation(
        self,
        signals: Dict[int, np.ndarray],
        reference_element: Optional[int] = None,
    ) -> CalibrationResult:
        """
        Calibrate using cross-correlation method.

        Uses ambient signals to estimate relative phase and amplitude
        between array elements through cross-correlation.

        Args:
            signals: Dict mapping element index to signal
            reference_element: Reference element index (uses config default if None)

        Returns:
            CalibrationResult with calibration data
        """
        start_time = time.time()
        self._state = CalibrationState.IN_PROGRESS

        ref_elem = reference_element or self._cal_config.reference_element

        if ref_elem not in signals:
            self._state = CalibrationState.FAILED
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.CORRELATION,
                reference_element=ref_elem,
                measurements=[],
                element_calibrations={},
                overall_confidence=0.0,
                timestamp=start_time,
                duration=time.time() - start_time,
                error_message=f"Reference element {ref_elem} not in signals",
            )

        reference_signal = signals[ref_elem]
        measurements = []
        element_calibrations = {}

        # Measure each element against reference
        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx not in signals:
                continue

            if idx == ref_elem:
                # Reference element has zero offset
                measurements.append(
                    CalibrationMeasurement(
                        element_index=idx,
                        reference_element=ref_elem,
                        phase_offset=0.0,
                        amplitude_ratio=1.0,
                        delay_samples=0.0,
                        timestamp=time.time(),
                        confidence=1.0,
                    )
                )
                element_calibrations[idx] = ElementCalibration(
                    element_index=idx,
                    phase_offset=0.0,
                    amplitude_scale=1.0,
                    delay_samples=0.0,
                    calibration_frequency=self._config.common_frequency,
                    calibration_timestamp=time.time(),
                )
                continue

            # Multiple measurements for averaging
            phase_measurements = []
            amp_measurements = []
            delay_measurements = []
            confidences = []

            for _ in range(self._cal_config.num_averages):
                corr_result = self._correlator.correlate(
                    reference_signal,
                    signals[idx],
                    ref_elem,
                    idx,
                    center_frequency=self._config.common_frequency,
                )

                phase_measurements.append(corr_result.phase_offset)
                delay_measurements.append(corr_result.delay_samples)
                confidences.append(corr_result.confidence)

                # Amplitude ratio
                ref_power = np.mean(np.abs(reference_signal) ** 2)
                elem_power = np.mean(np.abs(signals[idx]) ** 2)
                if elem_power > 0:
                    amp_ratio = np.sqrt(ref_power / elem_power)
                else:
                    amp_ratio = 1.0
                amp_measurements.append(amp_ratio)

            # Average measurements (circular mean for phase)
            avg_phase = np.angle(np.mean(np.exp(1j * np.array(phase_measurements))))
            avg_amplitude = np.mean(amp_measurements)
            avg_delay = np.mean(delay_measurements)
            avg_confidence = np.mean(confidences)

            measurement = CalibrationMeasurement(
                element_index=idx,
                reference_element=ref_elem,
                phase_offset=avg_phase,
                amplitude_ratio=avg_amplitude,
                delay_samples=avg_delay,
                timestamp=time.time(),
                confidence=avg_confidence,
            )
            measurements.append(measurement)

            element_calibrations[idx] = ElementCalibration(
                element_index=idx,
                phase_offset=avg_phase,
                amplitude_scale=avg_amplitude,
                delay_samples=avg_delay,
                calibration_frequency=self._config.common_frequency,
                calibration_timestamp=time.time(),
            )

        # Calculate overall confidence
        if measurements:
            overall_confidence = np.mean([m.confidence for m in measurements])
        else:
            overall_confidence = 0.0

        success = overall_confidence >= self._cal_config.min_confidence

        result = CalibrationResult(
            success=success,
            method=CalibrationMethod.CORRELATION,
            reference_element=ref_elem,
            measurements=measurements,
            element_calibrations=element_calibrations,
            overall_confidence=overall_confidence,
            timestamp=start_time,
            duration=time.time() - start_time,
        )

        self._state = CalibrationState.COMPLETED if success else CalibrationState.FAILED
        self._last_result = result
        self._measurement_history.append(result)

        return result

    def calibrate_known_source(
        self,
        signals: Dict[int, np.ndarray],
        source_azimuth: float,
        source_elevation: float = 0.0,
        reference_element: Optional[int] = None,
    ) -> CalibrationResult:
        """
        Calibrate using a known source at a known position.

        Computes expected phase delays based on geometry and compares
        with measured phases.

        Args:
            signals: Dict mapping element index to signal
            source_azimuth: Known source azimuth (radians)
            source_elevation: Known source elevation (radians)
            reference_element: Reference element index

        Returns:
            CalibrationResult with calibration data
        """
        start_time = time.time()
        self._state = CalibrationState.IN_PROGRESS

        ref_elem = reference_element or self._cal_config.reference_element

        if ref_elem not in signals:
            self._state = CalibrationState.FAILED
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_SOURCE,
                reference_element=ref_elem,
                measurements=[],
                element_calibrations={},
                overall_confidence=0.0,
                timestamp=start_time,
                duration=time.time() - start_time,
                error_message=f"Reference element {ref_elem} not in signals",
            )

        from .array_config import SPEED_OF_LIGHT

        wavelength = SPEED_OF_LIGHT / self._config.common_frequency
        k = 2 * np.pi / wavelength

        # Direction unit vector
        cos_el = np.cos(source_elevation)
        u = np.array([
            cos_el * np.sin(source_azimuth),
            cos_el * np.cos(source_azimuth),
            np.sin(source_elevation),
        ])

        # Get reference element position
        ref_element_obj = self._config.get_element_by_index(ref_elem)
        if ref_element_obj is None:
            self._state = CalibrationState.FAILED
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.KNOWN_SOURCE,
                reference_element=ref_elem,
                measurements=[],
                element_calibrations={},
                overall_confidence=0.0,
                timestamp=start_time,
                duration=time.time() - start_time,
                error_message=f"Reference element {ref_elem} not found",
            )

        ref_pos = ref_element_obj.position.to_array()

        reference_signal = signals[ref_elem]
        measurements = []
        element_calibrations = {}

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx not in signals:
                continue

            if idx == ref_elem:
                measurements.append(
                    CalibrationMeasurement(
                        element_index=idx,
                        reference_element=ref_elem,
                        phase_offset=0.0,
                        amplitude_ratio=1.0,
                        delay_samples=0.0,
                        timestamp=time.time(),
                        confidence=1.0,
                        source_azimuth=source_azimuth,
                        source_elevation=source_elevation,
                    )
                )
                element_calibrations[idx] = ElementCalibration(
                    element_index=idx,
                    phase_offset=0.0,
                    amplitude_scale=1.0,
                    calibration_frequency=self._config.common_frequency,
                    calibration_timestamp=time.time(),
                )
                continue

            # Compute expected phase difference from geometry
            elem_pos = elem.position.to_array()
            path_diff = np.dot(elem_pos - ref_pos, u)
            expected_phase = k * path_diff

            # Measure actual phase difference
            corr_result = self._correlator.correlate(
                reference_signal,
                signals[idx],
                ref_elem,
                idx,
            )

            # Calibration offset is measured - expected
            phase_offset = self._wrap_phase(corr_result.phase_offset - expected_phase)

            # Amplitude ratio
            ref_power = np.mean(np.abs(reference_signal) ** 2)
            elem_power = np.mean(np.abs(signals[idx]) ** 2)
            amp_ratio = np.sqrt(ref_power / elem_power) if elem_power > 0 else 1.0

            measurement = CalibrationMeasurement(
                element_index=idx,
                reference_element=ref_elem,
                phase_offset=phase_offset,
                amplitude_ratio=amp_ratio,
                delay_samples=corr_result.delay_samples,
                timestamp=time.time(),
                confidence=corr_result.confidence,
                source_azimuth=source_azimuth,
                source_elevation=source_elevation,
            )
            measurements.append(measurement)

            element_calibrations[idx] = ElementCalibration(
                element_index=idx,
                phase_offset=phase_offset,
                amplitude_scale=amp_ratio,
                delay_samples=corr_result.delay_samples,
                calibration_frequency=self._config.common_frequency,
                calibration_timestamp=time.time(),
            )

        overall_confidence = np.mean([m.confidence for m in measurements]) if measurements else 0.0
        success = overall_confidence >= self._cal_config.min_confidence

        result = CalibrationResult(
            success=success,
            method=CalibrationMethod.KNOWN_SOURCE,
            reference_element=ref_elem,
            measurements=measurements,
            element_calibrations=element_calibrations,
            overall_confidence=overall_confidence,
            timestamp=start_time,
            duration=time.time() - start_time,
        )

        self._state = CalibrationState.COMPLETED if success else CalibrationState.FAILED
        self._last_result = result
        self._measurement_history.append(result)

        return result

    def calibrate_pilot_tone(
        self,
        signals: Dict[int, np.ndarray],
        pilot_frequency: float,
        reference_element: Optional[int] = None,
    ) -> CalibrationResult:
        """
        Calibrate using a pilot tone present in the signals.

        Extracts phase at the pilot frequency for each element.

        Args:
            signals: Dict mapping element index to signal
            pilot_frequency: Pilot tone frequency (Hz)
            reference_element: Reference element index

        Returns:
            CalibrationResult with calibration data
        """
        start_time = time.time()
        self._state = CalibrationState.IN_PROGRESS

        ref_elem = reference_element or self._cal_config.reference_element

        if ref_elem not in signals:
            self._state = CalibrationState.FAILED
            return CalibrationResult(
                success=False,
                method=CalibrationMethod.PILOT_TONE,
                reference_element=ref_elem,
                measurements=[],
                element_calibrations={},
                overall_confidence=0.0,
                timestamp=start_time,
                duration=time.time() - start_time,
                error_message=f"Reference element {ref_elem} not in signals",
            )

        sample_rate = self._config.common_sample_rate
        measurements = []
        element_calibrations = {}

        # Extract phase at pilot frequency for each element
        pilot_phases = {}
        pilot_amplitudes = {}
        pilot_confidences = {}

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx not in signals:
                continue

            signal = signals[idx]
            n = len(signal)

            # FFT
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(n, 1 / sample_rate)

            # Find bin closest to pilot frequency
            pilot_bin = np.argmin(np.abs(freqs - pilot_frequency))

            # Extract phase and amplitude
            pilot_phases[idx] = np.angle(fft[pilot_bin])
            pilot_amplitudes[idx] = np.abs(fft[pilot_bin])

            # Confidence from pilot SNR
            noise_floor = np.median(np.abs(fft))
            if noise_floor > 0:
                snr = pilot_amplitudes[idx] / noise_floor
                pilot_confidences[idx] = min(1.0, snr / 10)
            else:
                pilot_confidences[idx] = 1.0

        # Compute offsets relative to reference
        ref_phase = pilot_phases.get(ref_elem, 0.0)
        ref_amplitude = pilot_amplitudes.get(ref_elem, 1.0)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx not in pilot_phases:
                continue

            if idx == ref_elem:
                phase_offset = 0.0
                amp_ratio = 1.0
            else:
                phase_offset = self._wrap_phase(pilot_phases[idx] - ref_phase)
                amp_ratio = ref_amplitude / pilot_amplitudes[idx] if pilot_amplitudes[idx] > 0 else 1.0

            measurement = CalibrationMeasurement(
                element_index=idx,
                reference_element=ref_elem,
                phase_offset=phase_offset,
                amplitude_ratio=amp_ratio,
                delay_samples=0.0,
                timestamp=time.time(),
                confidence=pilot_confidences.get(idx, 0.0),
            )
            measurements.append(measurement)

            element_calibrations[idx] = ElementCalibration(
                element_index=idx,
                phase_offset=phase_offset,
                amplitude_scale=amp_ratio,
                calibration_frequency=self._config.common_frequency,
                calibration_timestamp=time.time(),
            )

        overall_confidence = np.mean([m.confidence for m in measurements]) if measurements else 0.0
        success = overall_confidence >= self._cal_config.min_confidence

        result = CalibrationResult(
            success=success,
            method=CalibrationMethod.PILOT_TONE,
            reference_element=ref_elem,
            measurements=measurements,
            element_calibrations=element_calibrations,
            overall_confidence=overall_confidence,
            timestamp=start_time,
            duration=time.time() - start_time,
        )

        self._state = CalibrationState.COMPLETED if success else CalibrationState.FAILED
        self._last_result = result
        self._measurement_history.append(result)

        return result

    def apply_calibration(
        self,
        result: CalibrationResult,
        config: Optional[ArrayConfig] = None,
    ) -> ArrayConfig:
        """
        Apply calibration result to array configuration.

        Args:
            result: Calibration result to apply
            config: Array config to update (uses internal config if None)

        Returns:
            Updated array configuration
        """
        target_config = config or self._config

        for idx, cal in result.element_calibrations.items():
            element = target_config.get_element_by_index(idx)
            if element is not None:
                element.calibration = cal

        return target_config

    def verify_calibration(
        self,
        signals: Dict[int, np.ndarray],
        expected_azimuth: float,
        tolerance_deg: float = 5.0,
    ) -> Tuple[bool, float]:
        """
        Verify calibration by checking if signal direction is correct.

        Args:
            signals: Dict mapping element index to signal
            expected_azimuth: Expected signal direction (radians)
            tolerance_deg: Tolerance in degrees

        Returns:
            Tuple of (verification_passed, actual_azimuth)
        """
        from .doa import BeamscanDoA

        doa = BeamscanDoA(self._config, azimuth_resolution=1.0)
        result = doa.estimate(signals)

        error = abs(result.azimuth - expected_azimuth)
        error_deg = np.degrees(error)

        passed = error_deg <= tolerance_deg

        return passed, result.azimuth

    def track_calibration_drift(
        self,
        signals: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        """
        Track calibration drift since last calibration.

        Compares current phase measurements with stored calibration.

        Args:
            signals: Dict mapping element index to signal

        Returns:
            Dict mapping element index to phase drift (radians)
        """
        if self._last_result is None:
            return {}

        # Perform new correlation measurement
        ref_elem = self._last_result.reference_element
        if ref_elem not in signals:
            return {}

        reference_signal = signals[ref_elem]
        drift = {}

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx not in signals or idx == ref_elem:
                drift[idx] = 0.0
                continue

            # Measure current phase
            corr_result = self._correlator.correlate(
                reference_signal,
                signals[idx],
                ref_elem,
                idx,
            )

            # Compare with stored calibration
            stored_cal = self._last_result.element_calibrations.get(idx)
            if stored_cal:
                current_offset = corr_result.phase_offset
                stored_offset = stored_cal.phase_offset
                drift[idx] = self._wrap_phase(current_offset - stored_offset)
            else:
                drift[idx] = 0.0

        return drift

    def get_calibration_history(self) -> List[CalibrationResult]:
        """Get history of calibration results."""
        return self._measurement_history.copy()

    def clear_history(self) -> None:
        """Clear calibration history."""
        self._measurement_history.clear()

    def _wrap_phase(self, phase: float) -> float:
        """Wrap phase to [-pi, pi]."""
        return (phase + np.pi) % (2 * np.pi) - np.pi

    def export_calibration(self, result: CalibrationResult) -> Dict:
        """
        Export calibration result to dictionary.

        Args:
            result: Calibration result to export

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "success": result.success,
            "method": result.method.value,
            "reference_element": result.reference_element,
            "overall_confidence": result.overall_confidence,
            "timestamp": result.timestamp,
            "duration": result.duration,
            "elements": {
                idx: {
                    "phase_offset": cal.phase_offset,
                    "amplitude_scale": cal.amplitude_scale,
                    "delay_samples": cal.delay_samples,
                    "calibration_frequency": cal.calibration_frequency,
                }
                for idx, cal in result.element_calibrations.items()
            },
        }

    def import_calibration(self, data: Dict) -> CalibrationResult:
        """
        Import calibration from dictionary.

        Args:
            data: Dictionary with calibration data

        Returns:
            CalibrationResult
        """
        element_calibrations = {}
        for idx_str, cal_data in data.get("elements", {}).items():
            idx = int(idx_str)
            element_calibrations[idx] = ElementCalibration(
                element_index=idx,
                phase_offset=cal_data["phase_offset"],
                amplitude_scale=cal_data["amplitude_scale"],
                delay_samples=cal_data.get("delay_samples", 0.0),
                calibration_frequency=cal_data.get("calibration_frequency", 0.0),
                calibration_timestamp=data.get("timestamp", 0.0),
            )

        return CalibrationResult(
            success=data.get("success", True),
            method=CalibrationMethod(data.get("method", "correlation")),
            reference_element=data.get("reference_element", 0),
            measurements=[],
            element_calibrations=element_calibrations,
            overall_confidence=data.get("overall_confidence", 1.0),
            timestamp=data.get("timestamp", 0.0),
            duration=data.get("duration", 0.0),
        )

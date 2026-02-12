"""
Beamforming algorithms for antenna arrays.

Provides delay-and-sum beamforming and related spatial filtering
techniques for phased array signal processing.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .array_config import SPEED_OF_LIGHT, ArrayConfig, ElementPosition

logger = logging.getLogger(__name__)


class BeamformingMethod(Enum):
    """Beamforming algorithm types."""

    DELAY_AND_SUM = "delay_and_sum"  # Classic delay-and-sum
    PHASE_SHIFT = "phase_shift"  # Phase-shift only (narrowband)
    MVDR = "mvdr"  # Minimum Variance Distortionless Response


@dataclass
class SteeringVector:
    """Steering vector for a specific direction."""

    azimuth: float  # Azimuth angle in radians (0 = boresight)
    elevation: float  # Elevation angle in radians (0 = horizon)
    frequency: float  # Frequency in Hz
    weights: np.ndarray  # Complex weights for each element

    @property
    def azimuth_deg(self) -> float:
        """Azimuth in degrees."""
        return np.degrees(self.azimuth)

    @property
    def elevation_deg(self) -> float:
        """Elevation in degrees."""
        return np.degrees(self.elevation)


@dataclass
class BeamformerOutput:
    """Output from beamformer processing."""

    output_signal: np.ndarray  # Beamformed signal
    azimuth: float  # Steering azimuth in radians
    elevation: float  # Steering elevation in radians
    beam_power: float  # Output power in dB
    weights_used: np.ndarray  # Weights applied


@dataclass
class BeamPattern:
    """Beam pattern data for visualization."""

    azimuths: np.ndarray  # Azimuth angles in radians
    elevations: np.ndarray  # Elevation angles in radians
    pattern: np.ndarray  # Pattern values (dB)
    frequency: float  # Frequency in Hz
    peak_azimuth: float  # Azimuth of main beam peak
    peak_elevation: float  # Elevation of main beam peak
    beamwidth_az: float  # 3dB beamwidth in azimuth (radians)
    beamwidth_el: float  # 3dB beamwidth in elevation (radians)
    first_null_az: float  # First null position in azimuth (radians)


class Beamformer:
    """
    Beamformer for antenna array signal processing.

    Implements delay-and-sum and phase-shift beamforming for
    steering the array response toward a desired direction.

    Features:
        - Delay-and-sum beamforming (wideband)
        - Phase-shift beamforming (narrowband)
        - Steering vector computation
        - Beam pattern calculation
        - Null steering support
        - Array factor computation

    Example:
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        # Steer beam to 30 degrees azimuth
        output = beamformer.steer_and_sum(
            signals,
            azimuth=np.radians(30)
        )

        # Compute beam pattern
        pattern = beamformer.compute_pattern(
            azimuths=np.linspace(-np.pi/2, np.pi/2, 181)
        )
    """

    def __init__(
        self,
        config: ArrayConfig,
        method: BeamformingMethod = BeamformingMethod.PHASE_SHIFT,
    ) -> None:
        """
        Initialize beamformer.

        Args:
            config: Array configuration
            method: Beamforming method to use
        """
        self._config = config
        self._method = method

        # Cache element positions as matrix
        self._positions = config.get_position_matrix()
        self._num_elements = len(self._positions)

        # Precompute wavelength
        self._wavelength = SPEED_OF_LIGHT / config.common_frequency
        self._frequency = config.common_frequency

        # Cache for steering vectors
        self._steering_cache: Dict[Tuple[float, float], SteeringVector] = {}

    @property
    def config(self) -> ArrayConfig:
        """Get array configuration."""
        return self._config

    @property
    def num_elements(self) -> int:
        """Get number of array elements."""
        return self._num_elements

    @property
    def frequency(self) -> float:
        """Get operating frequency."""
        return self._frequency

    @frequency.setter
    def frequency(self, freq: float) -> None:
        """Set operating frequency and update wavelength."""
        self._frequency = freq
        self._wavelength = SPEED_OF_LIGHT / freq
        self._steering_cache.clear()  # Invalidate cache

    def compute_steering_vector(
        self,
        azimuth: float,
        elevation: float = 0.0,
        frequency: Optional[float] = None,
    ) -> SteeringVector:
        """
        Compute steering vector for a given direction.

        Args:
            azimuth: Azimuth angle in radians (0 = boresight, positive = right)
            elevation: Elevation angle in radians (0 = horizon, positive = up)
            frequency: Optional frequency override

        Returns:
            SteeringVector with complex weights
        """
        freq = frequency or self._frequency
        wavelength = SPEED_OF_LIGHT / freq

        # Check cache
        cache_key = (azimuth, elevation)
        if frequency is None and cache_key in self._steering_cache:
            return self._steering_cache[cache_key]

        # Direction unit vector (pointing toward source)
        # In our coordinate system: X=East, Y=North, Z=Up
        # Azimuth: 0=North (Y+), positive=clockwise when viewed from above
        # Elevation: 0=horizon, positive=up
        cos_el = np.cos(elevation)
        u = np.array(
            [
                cos_el * np.sin(azimuth),  # X component
                cos_el * np.cos(azimuth),  # Y component
                np.sin(elevation),  # Z component
            ]
        )

        # Compute phase delays for each element
        # Phase = 2*pi/lambda * (position dot direction)
        k = 2 * np.pi / wavelength
        phases = k * (self._positions @ u)

        # Steering vector: conjugate to steer beam toward source
        weights = np.exp(-1j * phases).astype(np.complex64)

        # Normalize
        weights = weights / np.sqrt(self._num_elements)

        sv = SteeringVector(
            azimuth=azimuth,
            elevation=elevation,
            frequency=freq,
            weights=weights,
        )

        # Cache if using default frequency
        if frequency is None:
            self._steering_cache[cache_key] = sv

        return sv

    def steer_and_sum(
        self,
        signals: Dict[int, np.ndarray],
        azimuth: float,
        elevation: float = 0.0,
        apply_calibration: bool = True,
    ) -> BeamformerOutput:
        """
        Apply beamforming to steer toward a direction.

        Args:
            signals: Dict mapping element index to signal
            azimuth: Steering azimuth in radians
            elevation: Steering elevation in radians
            apply_calibration: Whether to apply calibration corrections

        Returns:
            BeamformerOutput with beamformed signal
        """
        # Get steering vector
        sv = self.compute_steering_vector(azimuth, elevation)

        # Get calibration corrections
        if apply_calibration:
            cal_vector = self._config.get_calibration_vector()
        else:
            cal_vector = np.ones(self._num_elements, dtype=np.complex64)

        # Combine steering and calibration
        weights = sv.weights * cal_vector[: len(sv.weights)]

        # Find common length
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return BeamformerOutput(
                output_signal=np.array([], dtype=np.complex64),
                azimuth=azimuth,
                elevation=elevation,
                beam_power=-np.inf,
                weights_used=weights,
            )
        min_len = min(lengths)

        # Apply weights and sum
        output = np.zeros(min_len, dtype=np.complex64)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < len(weights):
                output += signals[idx][:min_len] * weights[idx]

        # Compute output power
        if len(output) > 0:
            beam_power = 10 * np.log10(np.mean(np.abs(output) ** 2) + 1e-20)
        else:
            beam_power = -np.inf

        return BeamformerOutput(
            output_signal=output,
            azimuth=azimuth,
            elevation=elevation,
            beam_power=beam_power,
            weights_used=weights,
        )

    def scan(
        self,
        signals: Dict[int, np.ndarray],
        azimuths: np.ndarray,
        elevation: float = 0.0,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Scan beam across multiple azimuths and find peak.

        Args:
            signals: Dict mapping element index to signal
            azimuths: Array of azimuth angles to scan (radians)
            elevation: Elevation angle (radians)

        Returns:
            Tuple of (powers, peak_azimuth, peak_power)
        """
        powers = np.zeros(len(azimuths))

        for i, az in enumerate(azimuths):
            result = self.steer_and_sum(signals, az, elevation)
            powers[i] = result.beam_power

        peak_idx = np.argmax(powers)
        peak_azimuth = azimuths[peak_idx]
        peak_power = powers[peak_idx]

        return powers, peak_azimuth, peak_power

    def compute_array_factor(
        self,
        azimuth: float,
        elevation: float = 0.0,
        steering_azimuth: float = 0.0,
        steering_elevation: float = 0.0,
    ) -> complex:
        """
        Compute array factor for a given observation and steering direction.

        Args:
            azimuth: Observation azimuth (radians)
            elevation: Observation elevation (radians)
            steering_azimuth: Beam steering azimuth (radians)
            steering_elevation: Beam steering elevation (radians)

        Returns:
            Complex array factor
        """
        # Steering vector
        sv = self.compute_steering_vector(steering_azimuth, steering_elevation)

        # Observation direction
        wavelength = self._wavelength
        k = 2 * np.pi / wavelength

        cos_el = np.cos(elevation)
        u = np.array(
            [
                cos_el * np.sin(azimuth),
                cos_el * np.cos(azimuth),
                np.sin(elevation),
            ]
        )

        # Phase at each element for observation direction
        obs_phases = k * (self._positions @ u)
        obs_vector = np.exp(1j * obs_phases)

        # Array factor = sum of steering * observation
        af = np.sum(sv.weights * obs_vector)

        return af

    def compute_pattern(
        self,
        azimuths: Optional[np.ndarray] = None,
        elevations: Optional[np.ndarray] = None,
        steering_azimuth: float = 0.0,
        steering_elevation: float = 0.0,
        normalize: bool = True,
    ) -> BeamPattern:
        """
        Compute beam pattern over specified angles.

        Args:
            azimuths: Azimuth angles (radians), default -90 to 90 degrees
            elevations: Elevation angles (radians), default 0
            steering_azimuth: Beam steering azimuth (radians)
            steering_elevation: Beam steering elevation (radians)
            normalize: Normalize pattern to 0 dB peak

        Returns:
            BeamPattern with pattern data
        """
        if azimuths is None:
            azimuths = np.linspace(-np.pi / 2, np.pi / 2, 181)
        if elevations is None:
            elevations = np.array([0.0])

        # Compute pattern
        pattern = np.zeros((len(azimuths), len(elevations)), dtype=np.float64)

        for i, az in enumerate(azimuths):
            for j, el in enumerate(elevations):
                af = self.compute_array_factor(az, el, steering_azimuth, steering_elevation)
                pattern[i, j] = np.abs(af) ** 2

        # Convert to dB
        pattern_db = 10 * np.log10(pattern + 1e-20)

        if normalize:
            pattern_db = pattern_db - np.max(pattern_db)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(pattern_db), pattern_db.shape)
        peak_azimuth = azimuths[peak_idx[0]]
        peak_elevation = elevations[peak_idx[1]] if len(elevations) > 1 else 0.0

        # Calculate beamwidth (3dB points)
        if len(elevations) == 1:
            # 1D pattern
            pattern_1d = pattern_db[:, 0]
            beamwidth_az = self._find_beamwidth(azimuths, pattern_1d)
            first_null_az = self._find_first_null(azimuths, pattern_1d)
            beamwidth_el = 0.0
        else:
            # 2D pattern - find beamwidth in principal planes
            az_cut = pattern_db[:, len(elevations) // 2]
            beamwidth_az = self._find_beamwidth(azimuths, az_cut)
            first_null_az = self._find_first_null(azimuths, az_cut)

            el_cut = pattern_db[len(azimuths) // 2, :]
            beamwidth_el = self._find_beamwidth(elevations, el_cut)

        return BeamPattern(
            azimuths=azimuths,
            elevations=elevations,
            pattern=pattern_db,
            frequency=self._frequency,
            peak_azimuth=peak_azimuth,
            peak_elevation=peak_elevation,
            beamwidth_az=beamwidth_az,
            beamwidth_el=beamwidth_el,
            first_null_az=first_null_az,
        )

    def _find_beamwidth(self, angles: np.ndarray, pattern_db: np.ndarray) -> float:
        """Find 3dB beamwidth."""
        peak_idx = np.argmax(pattern_db)
        peak_val = pattern_db[peak_idx]
        threshold = peak_val - 3.0

        # Find -3dB points on each side
        left_idx = peak_idx
        right_idx = peak_idx

        while left_idx > 0 and pattern_db[left_idx] > threshold:
            left_idx -= 1

        while right_idx < len(pattern_db) - 1 and pattern_db[right_idx] > threshold:
            right_idx += 1

        beamwidth = angles[right_idx] - angles[left_idx]
        return abs(beamwidth)

    def _find_first_null(self, angles: np.ndarray, pattern_db: np.ndarray) -> float:
        """Find first null from boresight."""
        peak_idx = np.argmax(pattern_db)

        # Search right of peak
        for i in range(peak_idx + 1, len(pattern_db) - 1):
            if pattern_db[i] < pattern_db[i - 1] and pattern_db[i] < pattern_db[i + 1]:
                return angles[i] - angles[peak_idx]

        return np.pi / 2  # Default if no null found

    def create_null(
        self,
        signals: Dict[int, np.ndarray],
        steering_azimuth: float,
        null_azimuth: float,
        steering_elevation: float = 0.0,
        null_elevation: float = 0.0,
    ) -> BeamformerOutput:
        """
        Create a beam with a null in a specific direction.

        Uses projection to remove interference from null direction
        while maintaining gain toward steering direction.

        Args:
            signals: Dict mapping element index to signal
            steering_azimuth: Desired signal direction (radians)
            null_azimuth: Interference direction for null (radians)
            steering_elevation: Steering elevation (radians)
            null_elevation: Null elevation (radians)

        Returns:
            BeamformerOutput with interference-suppressed signal
        """
        # Get steering vectors
        sv_signal = self.compute_steering_vector(steering_azimuth, steering_elevation)
        sv_null = self.compute_steering_vector(null_azimuth, null_elevation)

        # Project out null direction
        # w = w_signal - (w_signal . w_null*) * w_null / ||w_null||^2
        projection = np.vdot(sv_signal.weights, sv_null.weights)
        null_norm = np.vdot(sv_null.weights, sv_null.weights)

        if abs(null_norm) > 1e-10:
            weights = sv_signal.weights - (projection / null_norm) * sv_null.weights
        else:
            weights = sv_signal.weights

        # Normalize
        norm = np.linalg.norm(weights)
        if norm > 1e-10:
            weights = weights / norm

        # Apply weights
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return BeamformerOutput(
                output_signal=np.array([], dtype=np.complex64),
                azimuth=steering_azimuth,
                elevation=steering_elevation,
                beam_power=-np.inf,
                weights_used=weights,
            )
        min_len = min(lengths)

        output = np.zeros(min_len, dtype=np.complex64)
        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < len(weights):
                output += signals[idx][:min_len] * weights[idx]

        beam_power = 10 * np.log10(np.mean(np.abs(output) ** 2) + 1e-20)

        return BeamformerOutput(
            output_signal=output,
            azimuth=steering_azimuth,
            elevation=steering_elevation,
            beam_power=beam_power,
            weights_used=weights,
        )

    def spatial_spectrum(
        self,
        signals: Dict[int, np.ndarray],
        azimuths: Optional[np.ndarray] = None,
        elevation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial spectrum (power vs angle).

        Args:
            signals: Dict mapping element index to signal
            azimuths: Azimuth angles to scan (radians)
            elevation: Elevation angle (radians)

        Returns:
            Tuple of (azimuths, power_spectrum)
        """
        if azimuths is None:
            azimuths = np.linspace(-np.pi / 2, np.pi / 2, 181)

        powers, _, _ = self.scan(signals, azimuths, elevation)

        return azimuths, powers

    def get_element_weights(
        self, azimuth: float, elevation: float = 0.0
    ) -> Dict[int, complex]:
        """
        Get weights for each element for a given steering direction.

        Args:
            azimuth: Steering azimuth (radians)
            elevation: Steering elevation (radians)

        Returns:
            Dict mapping element index to complex weight
        """
        sv = self.compute_steering_vector(azimuth, elevation)
        cal_vector = self._config.get_calibration_vector()

        weights = {}
        for i, elem in enumerate(self._config.enabled_elements):
            if i < len(sv.weights):
                weights[elem.index] = sv.weights[i] * cal_vector[i]

        return weights

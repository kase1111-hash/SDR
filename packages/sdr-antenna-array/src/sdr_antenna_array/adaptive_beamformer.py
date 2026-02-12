"""
Adaptive beamforming algorithms for antenna arrays.

Provides advanced beamforming techniques including MVDR (Minimum Variance
Distortionless Response) / Capon beamformer for interference suppression.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .array_config import SPEED_OF_LIGHT, ArrayConfig
from .beamformer import Beamformer, BeamformerOutput, SteeringVector

logger = logging.getLogger(__name__)


class AdaptiveMethod(Enum):
    """Adaptive beamforming methods."""

    MVDR = "mvdr"  # Minimum Variance Distortionless Response (Capon)
    LCMV = "lcmv"  # Linearly Constrained Minimum Variance
    GSC = "gsc"  # Generalized Sidelobe Canceller
    FROST = "frost"  # Frost adaptive beamformer


@dataclass
class AdaptiveBeamformerState:
    """State of the adaptive beamformer."""

    weights: np.ndarray  # Current adaptive weights
    covariance_matrix: np.ndarray  # Estimated covariance matrix
    num_snapshots: int  # Number of snapshots used
    output_power: float  # Current output power
    sinr_estimate: float  # Estimated SINR in dB


@dataclass
class MVDRResult:
    """Result from MVDR beamformer."""

    output_signal: np.ndarray
    weights: np.ndarray
    output_power: float
    interference_nulls: List[float]  # Directions of nulls (radians)
    sinr_improvement: float  # SINR improvement over conventional (dB)


class AdaptiveBeamformer:
    """
    Adaptive beamformer with interference suppression.

    Implements MVDR (Minimum Variance Distortionless Response), also known
    as the Capon beamformer. Automatically places nulls toward interference
    sources while maintaining unity gain toward the desired signal.

    The MVDR weights are computed as:
        w = R^(-1) * a / (a^H * R^(-1) * a)

    where:
        R = sample covariance matrix
        a = steering vector toward desired direction

    Features:
        - Automatic interference suppression
        - Diagonal loading for robustness
        - Recursive covariance estimation
        - SINR optimization

    Example:
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        # Process with MVDR beamformer
        result = adaptive.mvdr(
            signals,
            desired_azimuth=np.radians(0),
        )
        print(f"SINR improvement: {result.sinr_improvement:.1f} dB")
    """

    def __init__(
        self,
        config: ArrayConfig,
        diagonal_loading: float = 0.01,
        forgetting_factor: float = 0.99,
    ) -> None:
        """
        Initialize adaptive beamformer.

        Args:
            config: Array configuration
            diagonal_loading: Regularization factor (fraction of trace)
            forgetting_factor: Exponential forgetting factor for covariance
        """
        self._config = config
        self._positions = config.get_position_matrix()
        self._num_elements = len(self._positions)
        self._wavelength = SPEED_OF_LIGHT / config.common_frequency
        self._frequency = config.common_frequency

        self._diagonal_loading = diagonal_loading
        self._forgetting_factor = forgetting_factor

        # State
        self._covariance: Optional[np.ndarray] = None
        self._num_snapshots = 0

        # Conventional beamformer for comparison
        self._conventional = Beamformer(config)

    @property
    def config(self) -> ArrayConfig:
        """Get array configuration."""
        return self._config

    @property
    def num_elements(self) -> int:
        """Get number of array elements."""
        return self._num_elements

    @property
    def diagonal_loading(self) -> float:
        """Get diagonal loading factor."""
        return self._diagonal_loading

    @diagonal_loading.setter
    def diagonal_loading(self, value: float) -> None:
        """Set diagonal loading factor."""
        if value < 0:
            raise ValueError("Diagonal loading must be non-negative")
        self._diagonal_loading = value

    def _compute_steering_vector(
        self, azimuth: float, elevation: float = 0.0
    ) -> np.ndarray:
        """Compute steering vector for given direction."""
        k = 2 * np.pi / self._wavelength

        cos_el = np.cos(elevation)
        u = np.array([
            cos_el * np.sin(azimuth),
            cos_el * np.cos(azimuth),
            np.sin(elevation),
        ])

        phases = k * (self._positions @ u)
        return np.exp(1j * phases).astype(np.complex128)

    def estimate_covariance(
        self,
        signals: Dict[int, np.ndarray],
        block_size: int = 256,
    ) -> np.ndarray:
        """
        Estimate sample covariance matrix from signals.

        Args:
            signals: Dict mapping element index to signal
            block_size: Block size for covariance estimation

        Returns:
            Estimated covariance matrix (N x N)
        """
        # Build data matrix
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return np.eye(self._num_elements, dtype=np.complex128)

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex128)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        # Block-based covariance estimation
        n_blocks = max(1, min_len // block_size)
        R = np.zeros((self._num_elements, self._num_elements), dtype=np.complex128)

        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block = data[:, start:end]
            R += block @ block.conj().T

        R /= (n_blocks * block_size)

        # Update running covariance with forgetting factor
        if self._covariance is None:
            self._covariance = R
        else:
            self._covariance = (
                self._forgetting_factor * self._covariance
                + (1 - self._forgetting_factor) * R
            )

        self._num_snapshots += n_blocks * block_size

        return self._covariance

    def _apply_diagonal_loading(self, R: np.ndarray) -> np.ndarray:
        """Apply diagonal loading for regularization."""
        trace = np.trace(R).real
        loading = self._diagonal_loading * trace / self._num_elements
        return R + loading * np.eye(self._num_elements, dtype=np.complex128)

    def compute_mvdr_weights(
        self,
        covariance: np.ndarray,
        steering_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Compute MVDR/Capon weights.

        Args:
            covariance: Sample covariance matrix
            steering_vector: Steering vector toward desired direction

        Returns:
            MVDR weight vector
        """
        # Apply diagonal loading
        R_loaded = self._apply_diagonal_loading(covariance)

        # Compute R^(-1)
        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using pseudo-inverse")
            R_inv = np.linalg.pinv(R_loaded)

        # MVDR weights: w = R^(-1) * a / (a^H * R^(-1) * a)
        numerator = R_inv @ steering_vector
        denominator = steering_vector.conj() @ R_inv @ steering_vector

        if abs(denominator) < 1e-20:
            logger.warning("MVDR denominator near zero, using conventional weights")
            return steering_vector / np.sqrt(self._num_elements)

        weights = numerator / denominator

        return weights

    def mvdr(
        self,
        signals: Dict[int, np.ndarray],
        desired_azimuth: float,
        desired_elevation: float = 0.0,
        block_size: int = 256,
    ) -> MVDRResult:
        """
        Apply MVDR beamformer.

        Args:
            signals: Dict mapping element index to signal
            desired_azimuth: Desired signal direction (radians)
            desired_elevation: Desired elevation (radians)
            block_size: Block size for covariance estimation

        Returns:
            MVDRResult with beamformed output
        """
        # Estimate covariance
        R = self.estimate_covariance(signals, block_size)

        # Compute steering vector
        a = self._compute_steering_vector(desired_azimuth, desired_elevation)

        # Compute MVDR weights
        weights = self.compute_mvdr_weights(R, a)

        # Build data matrix
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return MVDRResult(
                output_signal=np.array([], dtype=np.complex64),
                weights=weights,
                output_power=-np.inf,
                interference_nulls=[],
                sinr_improvement=0.0,
            )

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex128)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        # Apply weights
        output = weights.conj() @ data

        # Compute output power
        output_power = 10 * np.log10(np.mean(np.abs(output) ** 2) + 1e-20)

        # Find interference nulls by scanning pattern
        interference_nulls = self._find_nulls(weights, desired_azimuth)

        # Estimate SINR improvement
        conv_result = self._conventional.steer_and_sum(signals, desired_azimuth)
        sinr_improvement = output_power - conv_result.beam_power

        return MVDRResult(
            output_signal=output.astype(np.complex64),
            weights=weights,
            output_power=output_power,
            interference_nulls=interference_nulls,
            sinr_improvement=sinr_improvement,
        )

    def _find_nulls(
        self,
        weights: np.ndarray,
        main_beam_az: float,
        threshold_db: float = -20.0,
    ) -> List[float]:
        """Find null directions in the beam pattern."""
        azimuths = np.linspace(-np.pi / 2, np.pi / 2, 181)
        pattern = np.zeros(len(azimuths))

        for i, az in enumerate(azimuths):
            sv = self._compute_steering_vector(az)
            pattern[i] = np.abs(weights.conj() @ sv) ** 2

        # Convert to dB and normalize
        pattern_db = 10 * np.log10(pattern + 1e-20)
        pattern_db -= np.max(pattern_db)

        # Find local minima below threshold (nulls)
        nulls = []
        for i in range(1, len(pattern_db) - 1):
            if (
                pattern_db[i] < pattern_db[i - 1]
                and pattern_db[i] < pattern_db[i + 1]
                and pattern_db[i] < threshold_db
            ):
                # Exclude main beam region
                if abs(azimuths[i] - main_beam_az) > np.radians(10):
                    nulls.append(azimuths[i])

        return nulls

    def compute_mvdr_spectrum(
        self,
        signals: Dict[int, np.ndarray],
        azimuths: Optional[np.ndarray] = None,
        elevation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MVDR spatial spectrum (Capon spectrum).

        The MVDR spectrum provides higher resolution than conventional
        beamformer spectrum.

        Args:
            signals: Dict mapping element index to signal
            azimuths: Azimuth angles to scan (radians)
            elevation: Elevation angle (radians)

        Returns:
            Tuple of (azimuths, spectrum_db)
        """
        if azimuths is None:
            azimuths = np.linspace(-np.pi / 2, np.pi / 2, 181)

        # Estimate covariance
        R = self.estimate_covariance(signals)
        R_loaded = self._apply_diagonal_loading(R)

        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)

        # Compute MVDR spectrum
        spectrum = np.zeros(len(azimuths))

        for i, az in enumerate(azimuths):
            a = self._compute_steering_vector(az, elevation)
            # P_mvdr = 1 / (a^H * R^(-1) * a)
            denominator = a.conj() @ R_inv @ a
            if abs(denominator) > 1e-20:
                spectrum[i] = 1.0 / denominator.real
            else:
                spectrum[i] = 1e20

        spectrum_db = 10 * np.log10(spectrum + 1e-20)

        return azimuths, spectrum_db

    def lcmv(
        self,
        signals: Dict[int, np.ndarray],
        constraint_directions: List[Tuple[float, complex]],
        block_size: int = 256,
    ) -> BeamformerOutput:
        """
        Linearly Constrained Minimum Variance beamformer.

        Allows specifying multiple constraints (gains toward specific directions).

        Args:
            signals: Dict mapping element index to signal
            constraint_directions: List of (azimuth, desired_response) tuples
            block_size: Block size for covariance estimation

        Returns:
            BeamformerOutput with beamformed signal
        """
        if not constraint_directions:
            raise ValueError("At least one constraint required")

        # Estimate covariance
        R = self.estimate_covariance(signals, block_size)
        R_loaded = self._apply_diagonal_loading(R)

        try:
            R_inv = np.linalg.inv(R_loaded)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_loaded)

        # Build constraint matrix C and response vector f
        n_constraints = len(constraint_directions)
        C = np.zeros((self._num_elements, n_constraints), dtype=np.complex128)
        f = np.zeros(n_constraints, dtype=np.complex128)

        for i, (az, response) in enumerate(constraint_directions):
            C[:, i] = self._compute_steering_vector(az)
            f[i] = response

        # LCMV weights: w = R^(-1) * C * (C^H * R^(-1) * C)^(-1) * f
        R_inv_C = R_inv @ C
        inner = C.conj().T @ R_inv_C

        try:
            inner_inv = np.linalg.inv(inner)
        except np.linalg.LinAlgError:
            inner_inv = np.linalg.pinv(inner)

        weights = R_inv_C @ inner_inv @ f

        # Apply weights
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return BeamformerOutput(
                output_signal=np.array([], dtype=np.complex64),
                azimuth=constraint_directions[0][0],
                elevation=0.0,
                beam_power=-np.inf,
                weights_used=weights,
            )

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex128)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        output = weights.conj() @ data
        beam_power = 10 * np.log10(np.mean(np.abs(output) ** 2) + 1e-20)

        return BeamformerOutput(
            output_signal=output.astype(np.complex64),
            azimuth=constraint_directions[0][0],
            elevation=0.0,
            beam_power=beam_power,
            weights_used=weights,
        )

    def gsc(
        self,
        signals: Dict[int, np.ndarray],
        desired_azimuth: float,
        mu: float = 0.01,
        block_size: int = 64,
    ) -> BeamformerOutput:
        """
        Generalized Sidelobe Canceller (GSC) adaptive beamformer.

        Implements the GSC structure with LMS adaptation.

        Args:
            signals: Dict mapping element index to signal
            desired_azimuth: Desired signal direction (radians)
            mu: LMS step size
            block_size: Block size for processing

        Returns:
            BeamformerOutput with adaptively beamformed signal
        """
        # Build data matrix
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return BeamformerOutput(
                output_signal=np.array([], dtype=np.complex64),
                azimuth=desired_azimuth,
                elevation=0.0,
                beam_power=-np.inf,
                weights_used=np.zeros(self._num_elements),
            )

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex128)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        # Quiescent weight vector (conventional beamformer toward desired)
        w_q = self._compute_steering_vector(desired_azimuth)
        w_q = w_q / np.sqrt(self._num_elements)

        # Blocking matrix (orthogonal to w_q)
        # Simple construction: I - w_q * w_q^H (projects out desired direction)
        B = np.eye(self._num_elements) - np.outer(w_q, w_q.conj())

        # Ensure B spans the orthogonal complement
        # Take first N-1 columns of B (after SVD cleanup)
        U, s, _ = np.linalg.svd(B)
        # Keep columns corresponding to non-zero singular values
        n_blocking = np.sum(s > 1e-10)
        B = U[:, :n_blocking]

        # Adaptive weights for sidelobe canceller
        w_a = np.zeros(n_blocking, dtype=np.complex128)

        # Process in blocks with LMS
        n_blocks = min_len // block_size
        output_list = []

        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            x = data[:, start:end]

            # Upper branch: quiescent beamformer output
            d = w_q.conj() @ x

            # Lower branch: blocked signal through adaptive filter
            x_blocked = B.conj().T @ x  # n_blocking x block_size

            # Adaptive filter output
            y = w_a.conj() @ x_blocked

            # Error signal
            e = d - y

            # LMS weight update (averaged over block)
            for j in range(block_size):
                w_a = w_a + mu * x_blocked[:, j] * np.conj(e[j])

            output_list.append(e)

        if output_list:
            output = np.concatenate(output_list)
        else:
            output = np.array([], dtype=np.complex128)

        # Compute equivalent weights
        weights = w_q - B @ w_a

        beam_power = 10 * np.log10(np.mean(np.abs(output) ** 2) + 1e-20) if len(output) > 0 else -np.inf

        return BeamformerOutput(
            output_signal=output.astype(np.complex64),
            azimuth=desired_azimuth,
            elevation=0.0,
            beam_power=beam_power,
            weights_used=weights,
        )

    def reset(self) -> None:
        """Reset adaptive beamformer state."""
        self._covariance = None
        self._num_snapshots = 0

    def get_state(self) -> AdaptiveBeamformerState:
        """Get current adaptive beamformer state."""
        if self._covariance is None:
            cov = np.eye(self._num_elements, dtype=np.complex128)
            weights = np.ones(self._num_elements, dtype=np.complex128) / np.sqrt(
                self._num_elements
            )
        else:
            cov = self._covariance
            # Compute weights for boresight
            a = self._compute_steering_vector(0.0)
            weights = self.compute_mvdr_weights(cov, a)

        output_power = 0.0  # Would need signal to compute
        sinr_estimate = 0.0

        return AdaptiveBeamformerState(
            weights=weights,
            covariance_matrix=cov,
            num_snapshots=self._num_snapshots,
            output_power=output_power,
            sinr_estimate=sinr_estimate,
        )

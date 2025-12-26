"""
Constellation Diagram - I/Q signal visualization for modulation analysis.

Displays complex samples as points on I/Q plane, useful for
analyzing digital modulation types (PSK, QAM, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


class ModulationOverlay(Enum):
    """Predefined modulation constellation overlays."""
    NONE = "none"
    BPSK = "bpsk"       # 2 points
    QPSK = "qpsk"       # 4 points
    PSK8 = "8psk"       # 8 points
    QAM16 = "16qam"     # 16 points
    QAM64 = "64qam"     # 64 points
    QAM256 = "256qam"   # 256 points


@dataclass
class ConstellationPoint:
    """Ideal constellation point for overlay."""
    i: float
    q: float
    symbol: Optional[str] = None


@dataclass
class ConstellationStats:
    """Statistics for constellation analysis."""
    evm_percent: float = 0.0          # Error Vector Magnitude (%)
    evm_db: float = 0.0               # EVM in dB
    phase_error_deg: float = 0.0      # Mean phase error (degrees)
    magnitude_error: float = 0.0      # Mean magnitude error
    snr_estimate_db: float = 0.0      # Estimated SNR from constellation
    iq_offset: complex = 0j           # DC offset (I + jQ)
    iq_imbalance: float = 0.0         # I/Q gain imbalance (ratio)


@dataclass
class ConstellationResult:
    """Result from constellation processing."""
    i_data: np.ndarray               # I (real) component array
    q_data: np.ndarray               # Q (imaginary) component array
    num_points: int                  # Number of points
    stats: ConstellationStats        # Constellation statistics

    # Bounds for display
    i_min: float = -1.5
    i_max: float = 1.5
    q_min: float = -1.5
    q_max: float = 1.5


class ConstellationDisplay:
    """
    Constellation diagram display for I/Q samples.

    Visualizes complex samples as points on the I/Q plane,
    useful for analyzing modulation quality and type.

    Features:
    - Configurable point history (persistence)
    - Modulation overlay (ideal constellation points)
    - EVM and phase error calculation
    - Automatic normalization
    - DC offset tracking
    """

    # Predefined ideal constellations (normalized to unit circle/square)
    IDEAL_CONSTELLATIONS = {
        ModulationOverlay.BPSK: [
            ConstellationPoint(-1.0, 0.0, "0"),
            ConstellationPoint(1.0, 0.0, "1"),
        ],
        ModulationOverlay.QPSK: [
            ConstellationPoint(0.707, 0.707, "00"),
            ConstellationPoint(-0.707, 0.707, "01"),
            ConstellationPoint(-0.707, -0.707, "11"),
            ConstellationPoint(0.707, -0.707, "10"),
        ],
        ModulationOverlay.PSK8: [
            ConstellationPoint(np.cos(i * np.pi/4), np.sin(i * np.pi/4), f"{i}")
            for i in range(8)
        ],
        ModulationOverlay.QAM16: [
            ConstellationPoint(i * 2/3 - 1, q * 2/3 - 1, f"{(q*4+i):X}")
            for q in range(4) for i in range(4)
        ],
    }

    def __init__(
        self,
        max_points: int = 1024,
        persistence: int = 1,
        normalize: bool = True,
        overlay: ModulationOverlay = ModulationOverlay.NONE
    ):
        """
        Initialize constellation display.

        Args:
            max_points: Maximum points to display at once
            persistence: Number of frames to keep points visible (1 = no persistence)
            normalize: Auto-normalize points to unit magnitude
            overlay: Modulation overlay to display
        """
        self._max_points = max_points
        self._persistence = persistence
        self._normalize = normalize
        self._overlay = overlay

        # Point buffer - stores multiple frames if persistence > 1
        self._buffers: List[np.ndarray] = []

        # Reference constellation for EVM calculation
        self._reference: Optional[List[ConstellationPoint]] = None
        if overlay != ModulationOverlay.NONE:
            self._reference = self.IDEAL_CONSTELLATIONS.get(overlay)

    @property
    def max_points(self) -> int:
        """Get maximum display points."""
        return self._max_points

    @max_points.setter
    def max_points(self, value: int) -> None:
        """Set maximum display points."""
        self._max_points = value

    @property
    def persistence(self) -> int:
        """Get persistence (number of frames to retain)."""
        return self._persistence

    @persistence.setter
    def persistence(self, value: int) -> None:
        """Set persistence."""
        self._persistence = max(1, value)
        # Trim buffers if needed
        while len(self._buffers) > self._persistence:
            self._buffers.pop(0)

    @property
    def overlay(self) -> ModulationOverlay:
        """Get current modulation overlay."""
        return self._overlay

    @overlay.setter
    def overlay(self, value: ModulationOverlay) -> None:
        """Set modulation overlay."""
        self._overlay = value
        if value != ModulationOverlay.NONE:
            self._reference = self.IDEAL_CONSTELLATIONS.get(value)
        else:
            self._reference = None

    def get_overlay_points(self) -> List[ConstellationPoint]:
        """Get ideal constellation points for current overlay."""
        if self._overlay == ModulationOverlay.NONE:
            return []
        return self.IDEAL_CONSTELLATIONS.get(self._overlay, [])

    def update(self, samples: np.ndarray) -> None:
        """
        Update display with new samples.

        Args:
            samples: Complex I/Q samples
        """
        # Limit to max points
        if len(samples) > self._max_points:
            samples = samples[:self._max_points]

        # Normalize if enabled
        if self._normalize:
            max_mag = np.max(np.abs(samples))
            if max_mag > 1e-10:
                samples = samples / max_mag

        # Add to buffer
        self._buffers.append(samples.copy())

        # Remove old frames beyond persistence
        while len(self._buffers) > self._persistence:
            self._buffers.pop(0)

    def process(self, samples: Optional[np.ndarray] = None) -> ConstellationResult:
        """
        Process samples and return constellation data.

        Args:
            samples: Optional new samples. If None, uses buffered data.

        Returns:
            ConstellationResult with I/Q data and statistics
        """
        if samples is not None:
            self.update(samples)

        # Combine all buffered frames
        if self._buffers:
            all_points = np.concatenate(self._buffers)
        else:
            all_points = np.array([0j], dtype=np.complex64)

        # Extract I and Q
        i_data = np.real(all_points).astype(np.float32)
        q_data = np.imag(all_points).astype(np.float32)

        # Calculate statistics
        stats = self._calculate_stats(all_points)

        # Calculate bounds with margin
        margin = 0.1
        if len(all_points) > 0:
            i_min = float(np.min(i_data)) - margin
            i_max = float(np.max(i_data)) + margin
            q_min = float(np.min(q_data)) - margin
            q_max = float(np.max(q_data)) + margin
        else:
            i_min, i_max, q_min, q_max = -1.5, 1.5, -1.5, 1.5

        return ConstellationResult(
            i_data=i_data,
            q_data=q_data,
            num_points=len(all_points),
            stats=stats,
            i_min=i_min,
            i_max=i_max,
            q_min=q_min,
            q_max=q_max,
        )

    def _calculate_stats(self, points: np.ndarray) -> ConstellationStats:
        """Calculate constellation statistics."""
        if len(points) == 0:
            return ConstellationStats()

        # DC offset
        dc_offset = np.mean(points)

        # I/Q imbalance (ratio of I to Q standard deviations)
        i_std = np.std(np.real(points))
        q_std = np.std(np.imag(points))
        iq_imbalance = i_std / q_std if q_std > 1e-10 else 1.0

        # Mean magnitude and phase
        magnitudes = np.abs(points)
        phases = np.angle(points)

        mean_mag = np.mean(magnitudes)
        mag_error = np.std(magnitudes) / mean_mag if mean_mag > 1e-10 else 0.0

        # Phase error (relative to expected positions)
        # Without reference, use variance around mean
        phase_error_rad = np.std(phases)
        phase_error_deg = float(np.degrees(phase_error_rad))

        # EVM calculation
        if self._reference and len(self._reference) > 0:
            # Calculate EVM relative to reference constellation
            evm = self._calculate_evm(points)
        else:
            # Estimate EVM from signal variance
            # EVM ≈ (noise power / signal power)^0.5
            signal_power = np.mean(magnitudes**2)
            noise_power = np.var(points)
            evm = np.sqrt(noise_power / signal_power) if signal_power > 1e-10 else 0.0

        evm_percent = float(evm * 100)
        evm_db = float(20 * np.log10(evm + 1e-10))

        # SNR estimate from EVM: SNR ≈ -2 * EVM_dB (approximate)
        snr_estimate = -evm_db if evm_db < 0 else 0.0

        return ConstellationStats(
            evm_percent=evm_percent,
            evm_db=evm_db,
            phase_error_deg=phase_error_deg,
            magnitude_error=float(mag_error),
            snr_estimate_db=snr_estimate,
            iq_offset=dc_offset,
            iq_imbalance=float(iq_imbalance),
        )

    def _calculate_evm(self, points: np.ndarray) -> float:
        """Calculate EVM against reference constellation."""
        if not self._reference:
            return 0.0

        # Convert reference to array
        ref_points = np.array([complex(p.i, p.q) for p in self._reference])

        # For each received point, find nearest reference point
        errors = []
        for point in points:
            distances = np.abs(point - ref_points)
            min_error = np.min(distances)
            errors.append(min_error**2)

        # EVM = sqrt(mean error power / mean reference power)
        mean_error_power = np.mean(errors)
        mean_ref_power = np.mean(np.abs(ref_points)**2)

        return np.sqrt(mean_error_power / mean_ref_power) if mean_ref_power > 1e-10 else 0.0

    def clear(self) -> None:
        """Clear all buffered points."""
        self._buffers.clear()

    def get_decision_boundaries(self) -> List[Tuple[float, float, float, float]]:
        """
        Get decision boundary lines for current modulation overlay.

        Returns:
            List of (x1, y1, x2, y2) line segments
        """
        boundaries = []

        if self._overlay == ModulationOverlay.BPSK:
            # Vertical line at x=0
            boundaries.append((0, -1.5, 0, 1.5))

        elif self._overlay == ModulationOverlay.QPSK:
            # Cross at origin
            boundaries.append((-1.5, 0, 1.5, 0))  # Horizontal
            boundaries.append((0, -1.5, 0, 1.5))  # Vertical

        elif self._overlay == ModulationOverlay.PSK8:
            # 8 radial lines at 22.5 degree intervals
            for i in range(8):
                angle = (i + 0.5) * np.pi / 4
                boundaries.append((0, 0, 1.5 * np.cos(angle), 1.5 * np.sin(angle)))

        elif self._overlay == ModulationOverlay.QAM16:
            # Grid lines
            for v in [-0.333, 0.333]:
                boundaries.append((v, -1.5, v, 1.5))  # Vertical
                boundaries.append((-1.5, v, 1.5, v))  # Horizontal

        return boundaries

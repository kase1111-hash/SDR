"""
Signal demodulators for common modulation schemes.

Supports analog and digital modulation types:
- AM, FM, SSB (analog)
- ASK/OOK, FSK, PSK (digital)
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod


class ModulationType(Enum):
    """Modulation types."""
    # Analog
    AM = "am"
    FM = "fm"
    USB = "usb"
    LSB = "lsb"
    CW = "cw"
    # Digital
    OOK = "ook"
    ASK = "ask"
    FSK = "fsk"
    BPSK = "bpsk"
    QPSK = "qpsk"
    GFSK = "gfsk"


class Demodulator(ABC):
    """Abstract base class for demodulators."""

    def __init__(self, sample_rate: float):
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @abstractmethod
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate samples."""
        pass

    def reset(self) -> None:
        """Reset demodulator state."""
        pass


class AMDemodulator(Demodulator):
    """
    AM envelope demodulator.

    Uses envelope detection (magnitude) for AM demodulation.
    """

    def __init__(self, sample_rate: float, dc_block: bool = True):
        super().__init__(sample_rate)
        self._dc_block = dc_block
        self._dc_avg = 0.0
        self._dc_alpha = 0.001

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate AM signal."""
        # Envelope detection
        envelope = np.abs(samples)

        # DC blocking (remove carrier component)
        if self._dc_block:
            output = np.zeros_like(envelope)
            for i, s in enumerate(envelope):
                self._dc_avg = self._dc_alpha * s + (1 - self._dc_alpha) * self._dc_avg
                output[i] = s - self._dc_avg
            return output

        return envelope

    def reset(self) -> None:
        self._dc_avg = 0.0


class FMDemodulator(Demodulator):
    """
    FM demodulator using quadrature detection.

    Computes instantaneous frequency from phase differences.
    """

    def __init__(self, sample_rate: float, max_deviation: float = 75e3):
        super().__init__(sample_rate)
        self._max_deviation = max_deviation
        self._last_sample = 0.0 + 0.0j

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FM signal using quadrature detection."""
        # Prepend last sample for continuity
        extended = np.concatenate([[self._last_sample], samples])
        self._last_sample = samples[-1]

        # Quadrature demodulation
        # d(angle)/dt = Im(conj(x[n-1]) * x[n]) / |x[n-1]|^2
        delayed = extended[:-1]
        current = extended[1:]

        # Compute phase difference
        product = np.conj(delayed) * current
        phase_diff = np.angle(product)

        # Normalize by max deviation
        demod = phase_diff * (self._sample_rate / (2 * np.pi * self._max_deviation))

        return demod

    def reset(self) -> None:
        self._last_sample = 0.0 + 0.0j


class SSBDemodulator(Demodulator):
    """
    SSB (Single Sideband) demodulator.

    Supports USB (upper sideband) and LSB (lower sideband).
    """

    def __init__(self, sample_rate: float, mode: str = "usb"):
        super().__init__(sample_rate)
        self._mode = mode.lower()

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate SSB signal."""
        if self._mode == "lsb":
            # For LSB, conjugate to flip spectrum
            samples = np.conj(samples)

        # Extract real part (product detection with carrier)
        return samples.real


class OOKDemodulator(Demodulator):
    """
    OOK (On-Off Keying) demodulator.

    Simple threshold-based digital demodulation.
    """

    def __init__(self, sample_rate: float, threshold: float = 0.5):
        super().__init__(sample_rate)
        self._threshold = threshold
        self._auto_threshold = True

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate OOK signal to bits."""
        envelope = np.abs(samples)

        if self._auto_threshold:
            # Automatic threshold based on signal statistics
            threshold = (np.max(envelope) + np.min(envelope)) / 2
        else:
            threshold = self._threshold * np.max(envelope)

        return (envelope > threshold).astype(np.float32)

    def set_threshold(self, threshold: float, auto: bool = False) -> None:
        """Set demodulation threshold."""
        self._threshold = threshold
        self._auto_threshold = auto


class FSKDemodulator(Demodulator):
    """
    FSK (Frequency Shift Keying) demodulator.

    Uses quadrature detection to recover frequency shifts.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        deviation: float = 2400
    ):
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._deviation = deviation
        self._fm_demod = FMDemodulator(sample_rate, deviation)

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FSK signal."""
        # First, FM demodulate
        freq = self._fm_demod.demodulate(samples)

        # The output represents frequency deviation
        # Positive = mark, Negative = space (for standard FSK)
        return freq

    def demodulate_bits(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FSK to bits with symbol timing."""
        freq = self.demodulate(samples)

        # Simple slicing at threshold
        bits = (freq > 0).astype(np.float32)

        return bits

    def reset(self) -> None:
        self._fm_demod.reset()


class PSKDemodulator(Demodulator):
    """
    PSK (Phase Shift Keying) demodulator.

    Supports BPSK, QPSK, and higher-order PSK.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        order: int = 2  # 2=BPSK, 4=QPSK
    ):
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._order = order
        self._last_phase = 0.0

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate PSK signal to symbols."""
        # Extract phase
        phase = np.angle(samples)

        if self._order == 2:  # BPSK
            # Map to 0 or 1
            symbols = (phase > 0).astype(np.float32)
        elif self._order == 4:  # QPSK
            # Map to 0, 1, 2, 3
            symbols = np.floor((phase + np.pi) / (np.pi / 2)) % 4
        else:
            # General M-PSK
            symbols = np.floor((phase + np.pi) / (2 * np.pi / self._order)) % self._order

        return symbols

    def get_constellation(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get I/Q constellation points."""
        return samples.real, samples.imag


def create_demodulator(
    mod_type: ModulationType,
    sample_rate: float,
    **kwargs
) -> Demodulator:
    """
    Factory function to create demodulators.

    Args:
        mod_type: Modulation type
        sample_rate: Sample rate in Hz
        **kwargs: Additional demodulator-specific parameters

    Returns:
        Configured demodulator instance
    """
    if mod_type == ModulationType.AM:
        return AMDemodulator(sample_rate, **kwargs)
    elif mod_type == ModulationType.FM:
        return FMDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.USB, ModulationType.LSB):
        return SSBDemodulator(sample_rate, mod_type.value)
    elif mod_type in (ModulationType.OOK, ModulationType.ASK):
        return OOKDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.FSK, ModulationType.GFSK):
        return FSKDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.BPSK, ModulationType.QPSK):
        order = 2 if mod_type == ModulationType.BPSK else 4
        return PSKDemodulator(sample_rate, order=order, **kwargs)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

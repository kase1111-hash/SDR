"""
Base protocol decoder framework.

Provides abstract base classes and utilities for
implementing protocol-specific decoders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ProtocolType(Enum):
    """Protocol categories."""

    UNKNOWN = "unknown"
    ISM = "ism"
    AMATEUR = "amateur"
    AVIATION = "aviation"
    PAGING = "paging"
    TRUNKING = "trunking"
    IOT = "iot"
    BROADCAST = "broadcast"


@dataclass
class ProtocolInfo:
    """Protocol information and metadata."""

    name: str
    protocol_type: ProtocolType
    frequency_range: tuple  # (min_hz, max_hz)
    bandwidth_hz: float
    modulation: str
    symbol_rate: Optional[float] = None
    description: str = ""


@dataclass
class DecodedFrame:
    """Decoded protocol frame/packet."""

    protocol: str
    timestamp: float
    raw_bits: np.ndarray
    data: Dict[str, Any]
    is_valid: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtocolDecoder(ABC):
    """
    Abstract base class for protocol decoders.

    Subclass this to implement specific protocol decoders.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize decoder.

        Args:
            sample_rate: Sample rate of input signal
        """
        self._sample_rate = sample_rate
        self._info: Optional[ProtocolInfo] = None

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    @abstractmethod
    def protocol_info(self) -> ProtocolInfo:
        """Get protocol information."""
        pass

    @abstractmethod
    def decode(self, samples: np.ndarray) -> List[DecodedFrame]:
        """
        Decode samples to protocol frames.

        Args:
            samples: Complex I/Q samples

        Returns:
            List of decoded frames
        """
        pass

    @abstractmethod
    def can_decode(self, samples: np.ndarray) -> float:
        """
        Check if samples might contain this protocol.

        Args:
            samples: Complex I/Q samples

        Returns:
            Confidence score (0.0 to 1.0)
        """
        pass

    def reset(self) -> None:
        """Reset decoder state."""
        pass

    def _find_preamble(self, bits: np.ndarray, preamble: np.ndarray) -> List[int]:
        """
        Find preamble pattern in bit stream.

        Args:
            bits: Input bit stream
            preamble: Preamble pattern to find

        Returns:
            List of start indices where preamble found
        """
        indices = []
        preamble_len = len(preamble)

        for i in range(len(bits) - preamble_len + 1):
            if np.array_equal(bits[i : i + preamble_len], preamble):
                indices.append(i)

        return indices

    def _correlate_preamble(
        self, signal: np.ndarray, preamble: np.ndarray, threshold: float = 0.7
    ) -> List[int]:
        """
        Find preamble using correlation.

        Args:
            signal: Input signal
            preamble: Expected preamble signal
            threshold: Correlation threshold (0-1)

        Returns:
            List of indices where correlation exceeds threshold
        """
        correlation = np.correlate(signal, preamble, mode="valid")
        max_corr = np.max(np.abs(correlation))
        normalized = np.abs(correlation) / (max_corr + 1e-10)

        indices = np.where(normalized > threshold)[0]
        return indices.tolist()

    def _manchester_decode(self, bits: np.ndarray) -> np.ndarray:
        """
        Decode Manchester-encoded bits.

        Args:
            bits: Manchester-encoded bit stream

        Returns:
            Decoded bits
        """
        if len(bits) % 2 != 0:
            bits = bits[:-1]

        pairs = bits.reshape(-1, 2)
        # Manchester: 01 -> 0, 10 -> 1
        decoded = (pairs[:, 0] > pairs[:, 1]).astype(np.uint8)

        return decoded

    def _crc_check(
        self, data: np.ndarray, polynomial: int, initial: int = 0xFFFF
    ) -> int:
        """
        Calculate CRC.

        Args:
            data: Input data bytes
            polynomial: CRC polynomial
            initial: Initial CRC value

        Returns:
            CRC value
        """
        crc = initial

        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc <<= 1
            crc &= 0xFFFF

        return crc

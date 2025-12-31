"""
Protocol detector for automatic protocol identification.

Scans signals and attempts to match against known protocols.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from .base import DecodedFrame, ProtocolDecoder, ProtocolInfo


@dataclass
class ProtocolMatch:
    """Protocol detection result."""

    protocol_info: ProtocolInfo
    confidence: float
    decoder: ProtocolDecoder


class ProtocolDetector:
    """
    Automatic protocol detector.

    Maintains a registry of protocol decoders and attempts
    to identify protocols from signal characteristics.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize detector.

        Args:
            sample_rate: Sample rate in Hz
        """
        self._sample_rate = sample_rate
        self._decoders: Dict[str, ProtocolDecoder] = {}
        self._decoder_classes: Dict[str, Type[ProtocolDecoder]] = {}

    def register_decoder(self, name: str, decoder_class: Type[ProtocolDecoder]) -> None:
        """
        Register a protocol decoder class.

        Args:
            name: Protocol name
            decoder_class: Decoder class (not instance)
        """
        self._decoder_classes[name] = decoder_class

    def get_decoder(self, name: str) -> Optional[ProtocolDecoder]:
        """
        Get or create a decoder instance.

        Args:
            name: Protocol name

        Returns:
            Decoder instance or None if not found
        """
        if name in self._decoders:
            return self._decoders[name]

        if name in self._decoder_classes:
            decoder = self._decoder_classes[name](self._sample_rate)
            self._decoders[name] = decoder
            return decoder

        return None

    def detect(
        self, samples: np.ndarray, min_confidence: float = 0.5
    ) -> List[ProtocolMatch]:
        """
        Detect protocols in samples.

        Args:
            samples: Complex I/Q samples
            min_confidence: Minimum confidence threshold

        Returns:
            List of protocol matches sorted by confidence
        """
        matches = []

        for name, decoder_class in self._decoder_classes.items():
            decoder = self.get_decoder(name)
            if decoder is None:
                continue

            try:
                confidence = decoder.can_decode(samples)
                if confidence >= min_confidence:
                    matches.append(
                        ProtocolMatch(
                            protocol_info=decoder.protocol_info,
                            confidence=confidence,
                            decoder=decoder,
                        )
                    )
            except Exception:
                pass  # Skip decoders that fail

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)

        return matches

    def decode(
        self, samples: np.ndarray, protocol_name: Optional[str] = None
    ) -> List[DecodedFrame]:
        """
        Decode samples using specified or auto-detected protocol.

        Args:
            samples: Complex I/Q samples
            protocol_name: Specific protocol to use (auto-detect if None)

        Returns:
            List of decoded frames
        """
        if protocol_name:
            decoder = self.get_decoder(protocol_name)
            if decoder:
                return decoder.decode(samples)
            return []

        # Auto-detect
        matches = self.detect(samples)
        if not matches:
            return []

        # Try decoding with best match
        return matches[0].decoder.decode(samples)

    def detect_and_decode(
        self, samples: np.ndarray, min_confidence: float = 0.5
    ) -> Tuple[List[ProtocolMatch], List[DecodedFrame]]:
        """
        Detect protocols and decode all matching.

        Args:
            samples: Complex I/Q samples
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (matches, all decoded frames)
        """
        matches = self.detect(samples, min_confidence)

        all_frames = []
        for match in matches:
            try:
                frames = match.decoder.decode(samples)
                all_frames.extend(frames)
            except Exception:
                pass

        return matches, all_frames

    def list_protocols(self) -> List[str]:
        """List registered protocol names."""
        return list(self._decoder_classes.keys())

    def get_protocol_info(self, name: str) -> Optional[ProtocolInfo]:
        """Get protocol information by name."""
        decoder = self.get_decoder(name)
        if decoder:
            return decoder.protocol_info
        return None

    def reset_all(self) -> None:
        """Reset all decoder states."""
        for decoder in self._decoders.values():
            decoder.reset()

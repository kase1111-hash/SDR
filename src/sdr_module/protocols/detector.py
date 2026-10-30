"""
Protocol detector for automatic protocol identification.

Tries the real decoders on a stretch of I/Q and reports the protocols they
find. By default every live decoder of :mod:`sdr_module.dsp.live_decoders`
that works at the sample rate (POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS,
RDS, Morse code) is registered, wrapped as a
:class:`~sdr_module.protocols.base.ProtocolDecoder`: its ``can_decode``
runs the decoder's own synchronisation over the samples (Mode S preambles,
POCSAG and FLEX sync words, HDLC flags, the ACARS sync characters, RDS
check words, Morse timing) and scores what it found: 1.0 for messages that
passed their checks, 0.6 for frames found that failed them, 0.0 for
nothing.

The detector answers "what is in these samples" (the channel at
``offset_hz`` from their centre); it does not watch a band. Choosing a
protocol from where the receiver is tuned is the Decoder panel's "Auto".
"""

import logging
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from .base import DecodedFrame, ProtocolDecoder, ProtocolInfo, ProtocolType

logger = logging.getLogger(__name__)

# What a decoder may raise on input it can't take (a decoder registered
# without can_decode or protocol_info raises AttributeError).
_DECODE_ERRORS = (
    ValueError,
    IndexError,
    RuntimeError,
    AttributeError,
    NotImplementedError,
    TypeError,
)


@dataclass
class ProtocolMatch:
    """Protocol detection result."""

    protocol_info: ProtocolInfo
    confidence: float
    decoder: ProtocolDecoder


#: Live decoder name -> (type, frequency range Hz, bandwidth Hz, modulation,
#: symbol rate, description) of its ProtocolInfo.
_LIVE_INFO: Dict[
    str, Tuple[ProtocolType, Tuple[float, float], float, str, Optional[float], str]
] = {
    "POCSAG": (
        ProtocolType.PAGING,
        (138e6, 470e6),
        25e3,
        "FSK",
        1200.0,
        "Pager messages (512, 1200 and 2400 baud)",
    ),
    "FLEX": (
        ProtocolType.PAGING,
        (929e6, 932e6),
        25e3,
        "FSK",
        1600.0,
        "Motorola FLEX pager messages",
    ),
    "AX.25/APRS": (
        ProtocolType.AMATEUR,
        (144e6, 148e6),
        12.5e3,
        "AFSK (Bell 202) on FM",
        1200.0,
        "Packet radio and APRS",
    ),
    "ADS-B": (
        ProtocolType.AVIATION,
        (1090e6, 1090e6),
        2e6,
        "PPM",
        1e6,
        "Aircraft transponder replies (Mode S)",
    ),
    "ACARS": (
        ProtocolType.AVIATION,
        (129e6, 137e6),
        10e3,
        "MSK on AM",
        2400.0,
        "Aircraft data-link messages",
    ),
    "RDS": (
        ProtocolType.BROADCAST,
        (87.5e6, 108e6),
        200e3,
        "BPSK on a 57 kHz FM subcarrier",
        1187.5,
        "Station name and RadioText of FM broadcasts",
    ),
    "CW (Morse)": (
        ProtocolType.AMATEUR,
        (0.1e6, 6e9),
        500.0,
        "OOK (CW)",
        None,
        "Morse code",
    ),
}

#: Confidence when frames were found but failed their checks.
FAILED_CHECKS_CONFIDENCE = 0.6

#: Decoders whose messages carry no check: how much must decode before it
#: counts (a burst of anything keyed can read as a letter of Morse code).
_MIN_DECODED = {"CW (Morse)": 4}


class LiveProtocolDecoder(ProtocolDecoder):
    """A live decoder (:mod:`sdr_module.dsp.live_decoders`) as a
    :class:`ProtocolDecoder`; each call decodes the samples it is given
    from scratch. Make one per protocol with :func:`live_protocol_decoder`.
    """

    #: The live decoder's name ("ADS-B", "AX.25/APRS", ...).
    live_name = ""
    #: Where the channel is, from the centre of the samples (Hz).
    offset_hz = 0.0

    def __init__(self, sample_rate: float):
        super().__init__(sample_rate)
        kind, band, bandwidth, modulation, rate, text = _LIVE_INFO.get(
            self.live_name, (ProtocolType.UNKNOWN, (0.0, 0.0), 0.0, "", None, "")
        )
        self._info = ProtocolInfo(
            name=self.live_name,
            protocol_type=kind,
            frequency_range=band,
            bandwidth_hz=bandwidth,
            modulation=modulation,
            symbol_rate=rate,
            description=text,
        )

    @property
    def protocol_info(self) -> ProtocolInfo:
        assert self._info is not None
        return self._info

    def _run(self, samples: np.ndarray) -> Tuple[List[Any], Any]:
        """``(messages, status)`` of a fresh live decoder over ``samples``
        (ValueError when it can't work at this sample rate)."""
        from ..dsp.live_decoders import create_live_decoder

        decoder = create_live_decoder(self.live_name, self.sample_rate, self.offset_hz)
        if decoder.status.problem:
            raise ValueError(decoder.status.problem)
        messages = list(decoder.process(np.asarray(samples)))
        messages += list(decoder.flush())
        return messages, decoder.status

    def can_decode(self, samples: np.ndarray) -> float:
        """1.0 when messages passed their checks, 0.6 when frames were found
        that failed them, 0.0 for nothing (the decoder's own sync search).
        Morse code, which has no checks, needs a few characters."""
        messages, status = self._run(samples)
        need = _MIN_DECODED.get(self.live_name)
        if need is not None:
            return 1.0 if status.valid >= need else 0.0
        if status.valid or any(getattr(m, "valid", True) for m in messages):
            return 1.0
        if status.failed or messages:
            return FAILED_CHECKS_CONFIDENCE
        return 0.0

    def decode(self, samples: np.ndarray) -> List[DecodedFrame]:
        """The messages in ``samples`` as :class:`DecodedFrame` (``data``
        holds the message's fields)."""
        frames = []
        for msg in self._run(samples)[0]:
            raw = getattr(msg, "raw_bits", b"") or b""
            data: Dict[str, Any] = {}
            if is_dataclass(msg):
                for item in fields(msg):
                    if item.name not in ("raw_bits", "protocol"):
                        data[item.name] = getattr(msg, item.name)
            frames.append(
                DecodedFrame(
                    protocol=self.live_name,
                    timestamp=float(getattr(msg, "timestamp", 0.0) or 0.0),
                    raw_bits=np.frombuffer(bytes(raw), dtype=np.uint8),
                    data=data,
                    is_valid=bool(getattr(msg, "valid", True)),
                    error_message=str(getattr(msg, "error_message", "") or ""),
                )
            )
        return frames


def live_protocol_decoder(
    name: str, offset_hz: float = 0.0
) -> Type[LiveProtocolDecoder]:
    """A :class:`LiveProtocolDecoder` class for the live decoder ``name``
    listening ``offset_hz`` from the centre of the samples."""
    return type(
        f"Live{''.join(c for c in name if c.isalnum())}Decoder",
        (LiveProtocolDecoder,),
        {"live_name": name, "offset_hz": float(offset_hz)},
    )


class ProtocolDetector:
    """
    Automatic protocol detector.

    Maintains a registry of protocol decoders and tries each one on the
    samples (see the module docstring). With ``register_defaults`` (the
    default) every live decoder that works at ``sample_rate`` is registered.
    """

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float = 0.0,
        register_defaults: bool = True,
    ):
        """
        Initialize detector.

        Args:
            sample_rate: Sample rate in Hz
            offset_hz: Where the channel is, from the centre of the samples
            register_defaults: Register the live decoders
        """
        self._sample_rate = sample_rate
        self._offset_hz = float(offset_hz)
        self._decoders: Dict[str, ProtocolDecoder] = {}
        self._decoder_classes: Dict[str, Type[ProtocolDecoder]] = {}
        if register_defaults:
            self.register_defaults()

    def register_defaults(self) -> None:
        """Register the live decoders that work at this sample rate."""
        from ..dsp.live_decoders import LIVE_DECODERS

        for name, cls in LIVE_DECODERS.items():
            if self._sample_rate >= getattr(cls, "MIN_SAMPLE_RATE", 0.0):
                self.register_decoder(
                    name, live_protocol_decoder(name, self._offset_hz)
                )

    def register_decoder(self, name: str, decoder_class: Type[ProtocolDecoder]) -> None:
        """
        Register a protocol decoder class.

        Args:
            name: Protocol name
            decoder_class: Decoder class (not instance)
        """
        self._decoder_classes[name] = decoder_class
        self._decoders.pop(name, None)

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

        for name in list(self._decoder_classes):
            try:
                decoder = self.get_decoder(name)
                if decoder is None:
                    continue
                confidence = float(decoder.can_decode(samples))
                if confidence >= min_confidence:
                    matches.append(
                        ProtocolMatch(
                            protocol_info=decoder.protocol_info,
                            confidence=confidence,
                            decoder=decoder,
                        )
                    )
            except _DECODE_ERRORS as e:
                logger.debug(f"Decoder {name} failed during detection: {e}")

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
            except _DECODE_ERRORS as e:
                logger.debug(f"Protocol decoder {match.protocol_info.name} failed: {e}")

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

"""
The tuned FM station's RDS, for the main window.

While broadcast FM is being listened to (mode FM with a broadcast
deviation) and Radio > RDS Station Names is on, :class:`RdsListener`
decodes RDS in the background from the receiver's own FM channel (no
second channel filter, so it costs little: about 0.8 ms per 33 ms block at
2.4 MS/s), whatever the Decoder panel shows. The station's name goes in a
badge next to FREQ and on the radio tuner's station line; the badge's
tooltip has the RadioText and program type. Retuning forgets the station.

No Qt here; the window owns the widgets.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..dsp.live_decoders import RDSLiveDecoder

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_CHANNEL_RATE",
    "RdsListener",
    "badge_text",
    "badge_tooltip",
    "tuner_text",
]

#: The receiver's channel must be at least this fast for the 57 kHz
#: subcarrier (the FM multiplex is the discriminator's output).
MIN_CHANNEL_RATE = 192e3


class RdsListener:
    """RDS of the station on the receiver's broadcast FM channel."""

    def __init__(self) -> None:
        self._decoder: Optional[RDSLiveDecoder] = None
        self._tuned: Optional[float] = None
        self._failed = False

    @property
    def decoder(self) -> Optional[RDSLiveDecoder]:
        return self._decoder

    def reset(self) -> None:
        """Stop listening (the next block starts afresh)."""
        self._decoder = None
        self._tuned = None

    def station(self) -> Any:
        """What is known about the station (an RDSData), or None."""
        return None if self._decoder is None else self._decoder.station

    def process(self, channel: np.ndarray, channel_rate: float, tuned_hz: float) -> Any:
        """Decode a block of the receiver's FM channel (at 0 Hz, at
        ``channel_rate``) tuned to ``tuned_hz``; returns the station."""
        if channel_rate < MIN_CHANNEL_RATE or not len(channel):
            return self.station()
        decoder = self._decoder
        if decoder is None or decoder.sample_rate != float(channel_rate):
            decoder = self._decoder = RDSLiveDecoder(
                channel_rate, 0.0, channel_ready=True
            )
            self._tuned = tuned_hz
        elif tuned_hz != self._tuned:
            # Another station.
            self._tuned = tuned_hz
            decoder.channel_changed()
        try:
            decoder.process(channel)
        except Exception as exc:  # pragma: no cover - defensive
            if not self._failed:
                self._failed = True
                logger.warning("Background RDS decoding failed: %s", exc)
            return None
        return decoder.station


def badge_text(station: Any) -> str:
    """The station's name for the badge next to FREQ ("" for none yet)."""
    if station is None or not getattr(station, "ps_complete", False):
        return ""
    return str(getattr(station, "ps_name", "") or "").strip()


def badge_tooltip(station: Any) -> str:
    """Everything known about the station, for the badge's tooltip."""
    if station is None:
        return ""
    name = badge_text(station)
    pi_code = int(getattr(station, "pi_code", 0) or 0)
    lines = [f"RDS: {name}" + (f" (PI {pi_code:04X})" if pi_code else "")]
    pty = str(getattr(station, "pty_name", "") or "")
    if pty and pty != "None":
        lines.append(f"Program type: {pty}")
    text = str(getattr(station, "radio_text", "") or "").strip()
    if text and getattr(station, "radio_text_complete", False):
        lines.append(text)
    clock = str(getattr(station, "clock_time", "") or "")
    if clock:
        lines.append(f"Clock: {clock}")
    lines.append(
        "Decoded from the station's RDS while you listen to broadcast FM "
        "(Radio > RDS Station Names)."
    )
    return "\n".join(lines)


def tuner_text(station: Any) -> str:
    """The radio tuner's station line: the name and the RadioText."""
    name = badge_text(station)
    if not name:
        return ""
    text = str(getattr(station, "radio_text", "") or "").strip()
    if text and getattr(station, "radio_text_complete", False):
        return f"{name} · {text}"
    return name

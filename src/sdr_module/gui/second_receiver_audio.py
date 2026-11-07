"""
The Second Receiver's listening options, as the main window has them.

* **Stereo**: broadcast FM plays in stereo where the station sends it,
  through the main window's own stereo path (``_ReceiverChain`` with
  ``stereo`` set, :class:`~.stereo_audio.StereoPath`); its STEREO lamp says
  what a station is received as and why, as the main window's badge does
  (:func:`stereo_reading`).

What is decided here is Qt-free: the Second Receiver's DSP runs on a worker
thread (``second_receiver.FrameWork``), which hands the lamp's reading back
to the window with each frame.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

# The main window's thresholds: STEREO lights at half the stereo
# difference or more (about 10 dB of separation) and goes out below a
# third of it (no flicker while the blend hovers).
from .main_window import _STEREO_OFF_BLEND, _STEREO_ON_BLEND

__all__ = ["NO_STEREO_READING", "StereoReading", "stereo_reading"]

#: ``(text, tone, tooltip)`` of the STEREO lamp: "STEREO" or "MONO" (""
#: hides it), its theme tone (None: plain) and why.
StereoReading = Tuple[str, Optional[str], str]
#: No broadcast FM station is heard: no lamp.
NO_STEREO_READING: StereoReading = ("", None, "")


def stereo_reading(
    received: bool,
    enabled: bool,
    usable: bool,
    decoder: Any,
    lit: bool,
    switch: str = "the Stereo switch",
) -> StereoReading:
    """The STEREO lamp for a broadcast FM station: ``received`` (one is
    above the squelch and someone listens), ``enabled`` (the Stereo switch
    is on), ``usable`` (the channel holds the stereo signal:
    ``StereoPath.usable``), ``decoder`` (the chain's stereo decoder while it
    follows the audio, else None), ``lit`` (the lamp said STEREO for the
    last block: it goes out only below a third of the stereo difference).
    ``switch`` names the switch that turns stereo off. The wording is the
    main window's badge's (``SDRMainWindow._show_stereo``)."""
    if not received:
        return NO_STEREO_READING
    if not enabled:
        return "MONO", None, f"Mono: {switch} is off."
    if not usable:
        return (
            "MONO",
            None,
            "Mono: the Filter is too narrow for the stereo signal. Broadcast "
            "FM's 200 kHz filter holds it.",
        )
    if decoder is None:
        return "MONO", None, "Mono: listening for the station's stereo pilot."
    blend = float(decoder.blend)
    if blend >= _STEREO_ON_BLEND or (lit and blend >= _STEREO_OFF_BLEND):
        how = (
            "Stereo: the station's 19 kHz pilot is locked and left and right "
            "are fully apart."
            if blend >= 0.95
            else "Stereo, blended partly toward mono to keep the hiss down: the "
            "signal is a little weak."
        )
        return (
            "STEREO",
            "success",
            f"{how} {switch[0].upper()}{switch[1:]} turns it off.",
        )
    if not decoder.pilot:
        return (
            "MONO",
            None,
            "Mono: this station sends no stereo pilot (it broadcasts in mono), "
            "or it is too weak to hear one.",
        )
    if float(decoder.target) >= _STEREO_ON_BLEND:
        return "MONO", None, "Mono for a moment: stereo is coming in."
    return (
        "MONO",
        None,
        "Mono: the signal is too weak for clean stereo, so it plays in mono to "
        "keep the hiss down. Stereo comes back as the signal gets stronger.",
    )

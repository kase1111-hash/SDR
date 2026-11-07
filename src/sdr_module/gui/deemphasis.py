"""
Broadcast FM de-emphasis: the time constants offered (Radio > Audio >
De-emphasis) and the switch from one to the other while a station plays.

An FM broadcaster lifts the treble before it transmits (pre-emphasis) and
the receiver cuts it back by as much (de-emphasis), taking the hiss down
with it. Stations in the Americas and South Korea use a 75 µs time
constant, those in Europe, Australia and most of the rest of the world
50 µs. Heard with the other one, the treble is wrong: 3.3 dB too bright
at 10 kHz (a 75 µs station heard with 50 µs) or as much too dull.

The receiver chain (``main_window._ReceiverChain``) folds the de-emphasis
into its audio low-pass, an FIR, so another time constant is another set
of audio filters. :func:`carry_state` starts each new filter where the
old one is in the stream (the same input history, the same decimation
phase): it gives at once what it would have given all along. A
:class:`StageCrossfade` then plays the old filters' output into the new
ones' over :data:`CROSSFADE_S`, so nothing clicks; the stereo difference path
(``stereo_audio.StereoPath``) does the same from the same sample, so left
and right stay aligned.

Plain NumPy, no Qt.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dsp.fm_stereo import DEEMPHASIS_S

__all__ = [
    "DEEMPHASIS_CHOICES",
    "DEFAULT_DEEMPHASIS",
    "DEEMPHASIS_RANGE_S",
    "CROSSFADE_S",
    "deemphasis_seconds",
    "deemphasis_key",
    "deemphasis_note",
    "usable_deemphasis",
    "carry_state",
    "StageCrossfade",
]

#: Radio > Audio > De-emphasis: (key, label, tooltip); the key is the time
#: constant in microseconds, as it is saved (audio/deemphasis).
DEEMPHASIS_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    (
        "75",
        "75 µs (the Americas, South Korea)",
        "For FM broadcast stations in the Americas and South Korea, which "
        "lift their treble with a 75 µs time constant: it is cut back by as "
        "much, so music sounds as it was sent (the default)",
    ),
    (
        "50",
        "50 µs (Europe, Australia, most of the world)",
        "For FM broadcast stations in Europe, Australia and most of the rest "
        "of the world, which lift their treble with a 50 µs time constant. "
        "With the wrong one music sounds too bright (or too dull)",
    ),
)
#: Today's behaviour, and what a damaged setting gives.
DEFAULT_DEEMPHASIS = "75"
_SECONDS: Dict[str, float] = {"75": 75e-6, "50": 50e-6}
#: A chain takes a time constant in this range (s); anything else (not a
#: number, out of range) is :data:`~sdr_module.dsp.fm_stereo.DEEMPHASIS_S`.
DEEMPHASIS_RANGE_S = (10e-6, 500e-6)
#: A new time constant crossfades from the old filters over this long.
CROSSFADE_S = 0.01


def deemphasis_seconds(key: Any) -> Optional[float]:
    """The time constant (s) of a choice's key (``"75"``, ``"50"``), or
    None for anything that isn't one."""
    return _SECONDS.get(str(key).strip())


def deemphasis_note(key: str) -> str:
    """What the status bar says when ``key`` is chosen."""
    if str(key).strip() == "50":
        return (
            "De-emphasis 50 µs: broadcast FM as stations in Europe, Australia "
            "and most of the world send it."
        )
    return (
        "De-emphasis 75 µs: broadcast FM as stations in the Americas and South "
        "Korea send it."
    )


def deemphasis_key(seconds: Any) -> Optional[str]:
    """The choice whose time constant ``seconds`` is, or None."""
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return None
    for key, tau in _SECONDS.items():
        if math.isclose(value, tau, rel_tol=1e-9):
            return key
    return None


def usable_deemphasis(value: Any) -> float:
    """``value`` as a de-emphasis time constant (s): a finite number in
    :data:`DEEMPHASIS_RANGE_S`, else the default (75 µs)."""
    if isinstance(value, bool):
        return DEEMPHASIS_S
    try:
        tau = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEEMPHASIS_S
    low, high = DEEMPHASIS_RANGE_S
    if not math.isfinite(tau) or not low <= tau <= high:
        return DEEMPHASIS_S
    return tau


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #
def carry_state(old: Any, new: Any) -> None:
    """Start the FIR stage ``new`` where ``old`` is in the stream (both
    ``main_window._FirDecimator``, with the same decimation factor): the
    same decimation phase, and the last input samples as its history (as
    many as its taps need; zeros before what ``old`` kept). Its next
    output is then what it would have been had it run all along."""
    new._phase = old._phase
    history = old._history
    taps = new._taps
    if history is None or taps is None:
        new._history = None  # (nothing yet, or no filter: none needed)
        return
    need = len(taps) - 1
    if len(history) >= need:
        new._history = history[len(history) - need :].copy()
    else:
        pad = np.zeros(need - len(history), dtype=history.dtype)
        new._history = np.concatenate((pad, history))


class StageCrossfade:
    """From one list of audio stages to another: both filter the same
    input for ``length`` output samples, and the output crossfades (a
    raised cosine) from the old stages' to the new ones'. The old stages
    are left behind once :attr:`finished`; the new ones carry on.

    Two of these started on the same block with the same ``length`` (the
    mono program and the stereo difference) ramp sample for sample alike,
    so left = mono + difference stays the crossfade of its own old and new
    filtering.
    """

    def __init__(self, old: Sequence[Any], new: Sequence[Any], length: int) -> None:
        self.old: List[Any] = list(old)
        self.new: List[Any] = list(new)
        self.length = max(1, int(length))
        self.done = 0  # output samples so far

    @property
    def finished(self) -> bool:
        return self.done >= self.length

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a block through both, crossfading (float32 out)."""
        before = x
        for stage in self.old:
            before = stage.process(before)
        after = x
        for stage in self.new:
            after = stage.process(after)
        n = min(len(before), len(after))
        if n == 0:
            return np.asarray(after, dtype=np.float32)
        position = np.minimum((self.done + np.arange(n) + 0.5) / self.length, 1.0)
        ramp = 0.5 - 0.5 * np.cos(np.pi * position)
        out = before[:n] + ramp * (after[:n] - before[:n])
        if len(after) > n:  # (never: the same factors and phases)
            out = np.concatenate((out, after[n:]))
        self.done += n
        return np.asarray(out, dtype=np.float32)

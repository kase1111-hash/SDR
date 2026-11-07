"""
One-channel (real) recordings read as I/Q.

A one-channel recording holds a real signal: audio from a receiver, or a
receiver's IF or RF sampled directly by a real ADC. Its spectrum from 0 Hz
to half its rate is all it holds (the negative frequencies only mirror it).
Taking the samples as I with Q = 0 would show every signal twice, at its
frequency and mirrored below 0. So it is read the way a receiver turns a
real IF into I/Q: as the analytic signal at half the rate, centred on a
quarter of the rate (:func:`iq_rate`, :func:`iq_center`). The I/Q then
holds the file's own frequencies, 0 Hz to half its rate, each once, and
every receiver mode works on them as on any I/Q recording.

How: the real samples are mixed down by a quarter of their rate (times
``(-j)**n``), low-passed to half the band and taken every other sample.
With the mix, the half-band filter's zeros and the decimation, I/Q sample
``m`` is ``(-1)**m * (x[2m] + j*H(x)[2m])``: I is every other real sample
with alternating sign, Q the samples between through a windowed Hilbert
filter (:data:`TAPS`, 2/(pi*d) for odd d, a Kaiser window). Over the inner
80% of the span (the file's 5% to 45% of its rate: what the receiver tunes)
its gain is flat to 0.0001 dB and the mirror image of a signal is at least
100 dB down. Nearer the edges (the file's lowest and highest frequencies,
below 5% or above 45% of its rate) the image grows and the signal weakens:
54 dB down at 4% (or 46%), 20 dB at 2%, 9 dB at 1% (the signal 2.6 dB
weaker there). The image of a signal near one edge shows near the other,
not on the signal: a Morse tone at 700 Hz in 48 kHz audio still decodes.

The filter is centred (no delay): I/Q sample ``m`` depends only on the
real samples around ``2m`` (:data:`CONTEXT` on each side; zeros before the
file's first sample and after its last). So any stretch of the file reads
the same by itself as within the whole (seeking needs no history).
NumPy only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

#: Real samples the filter reaches on each side of the one it centres on.
CONTEXT = 33
#: The Kaiser window's beta: over 100 dB of image rejection with 34 taps.
_BETA = 10.25


def _hilbert_taps() -> np.ndarray:
    """The Hilbert filter on the odd real samples, as :func:`real_to_iq`
    applies it (correlated with them, from offset +CONTEXT down to
    -CONTEXT)."""
    d = np.arange(-CONTEXT, CONTEXT + 1)
    window = np.kaiser(2 * CONTEXT + 1, _BETA)
    taps = np.zeros(2 * CONTEXT + 1)
    odd = d % 2 != 0
    taps[odd] = 2.0 / (np.pi * d[odd]) * window[odd]
    # (Correlation runs from the oldest sample: offset +CONTEXT first.)
    return taps[odd][::-1].astype(np.float32)


#: The odd taps of the Hilbert filter (34), in the order they are used.
TAPS = _hilbert_taps()


def iq_rate(frame_rate: float) -> float:
    """The I/Q sample rate of a real recording at ``frame_rate``: half."""
    return float(frame_rate) / 2.0


def iq_center(frame_rate: float) -> float:
    """The frequency in a real recording that the I/Q is centred on (0 Hz
    of the I/Q): a quarter of its rate, the middle of 0 to half the rate."""
    return float(frame_rate) / 4.0


def iq_count(frames: int) -> int:
    """I/Q samples a real recording of ``frames`` samples gives."""
    return max(0, int(frames)) // 2


def frame_span(first: int, count: int) -> Tuple[int, int]:
    """The real samples (``start``, ``stop``: frame indices, ``stop`` past
    the last) that I/Q samples ``first`` to ``first + count - 1`` need.
    ``start`` is below 0 near the beginning and ``stop`` may lie past the
    end: those samples are zeros."""
    first, count = int(first), max(0, int(count))
    return 2 * first - CONTEXT, 2 * (first + count) + CONTEXT - 1


def real_to_iq(frames: np.ndarray, first: int) -> np.ndarray:
    """I/Q samples from ``first`` on, as complex64, from the real samples
    ``frames`` of :func:`frame_span` (``2 * count + 2 * CONTEXT - 1`` of
    them, for ``count`` I/Q samples; any float or integer type, full scale
    1.0 gives full scale 1.0)."""
    x = np.asarray(frames, dtype=np.float32)
    count = (len(x) - 2 * CONTEXT + 1) // 2
    out = np.empty(max(0, count), dtype=np.complex64)
    if count <= 0:
        return out
    parts = out.view(np.float32)
    # I: the samples at 2m themselves. Q: the odd samples around them
    # through the Hilbert filter (the even samples' taps are zero).
    parts[0::2] = x[CONTEXT : CONTEXT + 2 * count : 2]
    parts[1::2] = np.correlate(x[0 : 2 * (count + CONTEXT) : 2], TAPS, "valid")
    # The quarter-rate mix: every other I/Q sample turned over.
    turned = out[(int(first) & 1) ^ 1 :: 2]
    np.negative(turned, out=turned)
    return out

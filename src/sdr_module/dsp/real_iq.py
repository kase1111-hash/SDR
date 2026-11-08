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
filter (:func:`hilbert_taps`, 2/(pi*d) for odd d, a Kaiser window).

Over the inner 80% of the span (the file's 5% to 45% of its rate: what the
receiver tunes) its gain is flat to 0.0001 dB and the mirror image of a
signal is at least 100 dB down. Nearer the edges (the file's lowest and
highest frequencies) the image grows and the signal weakens. With the
shortest filter (:data:`CONTEXT`, 34 taps, used at high rates) the image is
54 dB down at 4% of the rate (and at 46%), 20 dB at 2% (48%) and 9 dB at 1%
(49%). At lower rates, where it costs little, the filter is longer
(:func:`context_for`: 212 taps at 48 kHz), so in audio the image of a tone
is 54 dB down from 300 Hz up (108 dB at 700 Hz), within a budget of work
per second of signal. The image of a signal near one edge shows near the
other, not on the signal.

The filter is centred (no delay): I/Q sample ``m`` depends only on the
real samples around ``2m`` (``context`` on each side; zeros before the
file's first sample and after its last). So any stretch of the file reads
the same by itself as within the whole (seeking needs no history).
NumPy only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

#: Real samples the shortest filter reaches on each side of the one it
#: centres on (34 taps: the image 54 dB down at 4% of the rate).
CONTEXT = 33
#: The Kaiser window's beta: over 100 dB of image rejection in the inner
#: 80% of the span, whatever the filter's length.
_BETA = 10.25
#: Where the filter's length allows, a tone's image is 54 dB down from
#: this frequency up (the shortest filter's 4% of the rate, moved down).
LOW_HZ = 300.0
#: The most multiply-adds the filter may take per second of signal (the
#: shortest filter takes more only above 3 MS/s, where it is needed).
WORK_PER_S = 5e7
_MAX_CONTEXT = 1023


def context_for(frame_rate: float) -> int:
    """How far the filter reaches (real samples on each side, odd) for a
    recording at ``frame_rate``: at least :data:`CONTEXT`, longer at low
    rates so a tone's image is 54 dB down from :data:`LOW_HZ` up, within
    :data:`WORK_PER_S` multiply-adds a second (48 kHz: 211; 2.4 MHz: 39;
    from 3 MHz: 33)."""
    rate = max(1.0, float(frame_rate))
    wanted = CONTEXT * (0.04 * rate / LOW_HZ)
    affordable = WORK_PER_S / (rate / 2.0) - 1.0  # (context + 1 taps)
    context = int(max(CONTEXT, min(wanted, affordable, _MAX_CONTEXT)))
    # (Odd, the outermost taps being the odd ones: rounded down, so never
    # past the budget; CONTEXT itself is odd.)
    return context if context % 2 else context - 1


@lru_cache(maxsize=16)
def hilbert_taps(context: int = CONTEXT) -> np.ndarray:
    """The Hilbert filter on the odd real samples (``context + 1`` taps),
    in the order :func:`real_to_iq` applies them: correlated with the
    samples, from offset ``+context`` down to ``-context``. Read-only."""
    d = np.arange(-context, context + 1)
    window = np.kaiser(2 * context + 1, _BETA)
    taps = np.zeros(2 * context + 1)
    odd = d % 2 != 0
    taps[odd] = 2.0 / (np.pi * d[odd]) * window[odd]
    # (Correlation runs from the oldest sample: offset +context first.)
    out = taps[odd][::-1].astype(np.float32)
    out.setflags(write=False)
    return out


#: The shortest filter's taps (34).
TAPS = hilbert_taps(CONTEXT)

#: Up to this rate (Hz) a one-channel recording is probably audio, a
#: receiver's sound (an IF or RF sampled directly is faster): File > Decode
#: Recording and ``sdr-scan decode`` decode its sound as a receiver's audio
#: (unless its metadata places it elsewhere: :func:`is_audio`).
AUDIO_RATE_MAX = 200e3


def iq_rate(frame_rate: float) -> float:
    """The I/Q sample rate of a real recording at ``frame_rate``: half."""
    return float(frame_rate) / 2.0


def iq_center(frame_rate: float) -> float:
    """The frequency in a real recording that the I/Q is centred on (0 Hz
    of the I/Q): a quarter of its rate, the middle of 0 to half the rate."""
    return float(frame_rate) / 4.0


def is_audio(frame_rate: float, center_hz: float = 0.0) -> bool:
    """Whether a one-channel recording whose own rate is ``frame_rate``,
    centred on ``center_hz`` (0: nothing places it), is probably a
    receiver's audio, its sound, rather than an IF or RF sampled directly:
    an audio rate (up to :data:`AUDIO_RATE_MAX`) at its own frequencies (0
    Hz to half its rate, centred on a quarter of it). One its metadata
    places elsewhere (an IF at 10.7 MHz) holds that IF, decoded from its
    I/Q there. The window's File menu (Open, Recording Info, Convert and
    Decode Recording) and ``sdr-scan decode`` go by this one rule."""
    rate = float(frame_rate or 0.0)
    if not 0 < rate <= AUDIO_RATE_MAX:
        return False
    center = float(center_hz or 0.0)
    return not center or abs(center - iq_center(rate)) <= 0.5


def iq_count(frames: int) -> int:
    """I/Q samples a real recording of ``frames`` samples gives."""
    return max(0, int(frames)) // 2


def frame_span(first: int, count: int, context: int = CONTEXT) -> Tuple[int, int]:
    """The real samples (``start``, ``stop``: frame indices, ``stop`` past
    the last) that I/Q samples ``first`` to ``first + count - 1`` need with
    a filter reaching ``context``. ``start`` is below 0 near the beginning
    and ``stop`` may lie past the end: those samples are zeros."""
    first, count = int(first), max(0, int(count))
    return 2 * first - context, 2 * (first + count) + context - 1


def real_to_iq(frames: np.ndarray, first: int, context: int = CONTEXT) -> np.ndarray:
    """I/Q samples from ``first`` on, as complex64, from the real samples
    ``frames`` of :func:`frame_span` (``2 * count + 2 * context - 1`` of
    them, for ``count`` I/Q samples; any float or integer type, full scale
    1.0 gives full scale 1.0)."""
    x = np.asarray(frames, dtype=np.float32)
    count = (len(x) - 2 * context + 1) // 2
    out = np.empty(max(0, count), dtype=np.complex64)
    if count <= 0:
        return out
    parts = out.view(np.float32)
    # I: the samples at 2m themselves. Q: the odd samples around them
    # through the Hilbert filter (the even samples' taps are zero).
    parts[0::2] = x[context : context + 2 * count : 2]
    taps = hilbert_taps(context)
    parts[1::2] = np.correlate(x[0 : 2 * (count + context) : 2], taps, "valid")
    # The quarter-rate mix: every other I/Q sample turned over.
    turned = out[(int(first) & 1) ^ 1 :: 2]
    np.negative(turned, out=turned)
    return out

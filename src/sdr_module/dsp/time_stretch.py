"""
Time-stretch: audio played faster or slower with its pitch kept.

:class:`TimeStretch` is WSOLA (waveform-similarity overlap-add; Verhelst
and Roelands, 1993), streamed a block at a time. It is what plays a
recording's sound at 0.5x, 2x or 4x (``gui/playback_sound.py``).

* The output is made of frames of the input (``FRAME_S``, 25 ms) laid
  half a frame (a hop) apart, each faded in and out by a periodic Hann
  window whose halves add up to exactly one: every output sample is a
  weighted sum of input samples, so the output never steps.
* Each frame comes from its place in the input at the ratio (``ratio``
  input samples per output sample: 2 plays twice as fast), moved by up to
  ``SEARCH_S`` (7 ms, half the period of a 70 Hz voice) to where the
  input best continues what was just played (the normalized
  cross-correlation over the half frame they share). The waveforms line
  up where frames meet, so a tone or a voice keeps its pitch to a
  fraction of a cent, and nothing beats or warbles.
* It streams: :meth:`feed` the input as it comes, :meth:`pull` the output
  as it is wanted. A frame is made only once all the input it may use has
  come, so the output doesn't depend on how the input was cut into blocks
  (any block sizes give the same audio): nothing clicks where blocks meet.
  The ratio may change from one pull to the next; each frame is made with
  the ratio of the pull that makes it.
* Mono (1-D) or stereo (``(n, 2)``: left, right) blocks: both channels
  take each frame from the same place (found on their sum), so the stereo
  image stays as it was. A stream turns stereo at its first stereo block
  (its mono plays on both channels from then on) until :meth:`reset`.
* The first frame starts at full level (no fade-in of its own), so a
  stream taken over mid-sound continues it without a dip; :meth:`finish`
  makes the rest of the input into output and fades out (the stream
  ended), and :meth:`drain` hands the rest back as it is: the output then
  continues the input exactly, so blocks played as they are join it
  without a step.

NumPy only: two real FFTs of 2048 points and an inverse one per hop of
output (12.5 ms), so the work follows the time played, not the speed
(about 0.2 ms per 33 ms display frame in mono, 0.3 ms in stereo, on a
4-core computer).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

#: Frame length (seconds): about the shortest that keeps a low voice's
#: harmonics clean (at 20 ms a 90 Hz voice gets rough), and short enough
#: that speech at 4x still keeps a piece of every sound.
FRAME_S = 0.025
#: How far a frame may move from its place to line up with the sound just
#: played (seconds either way): half the period of a 70 Hz voice. At 4 ms
#: a 90 Hz voice lost its pitch (its frames could not line up).
SEARCH_S = 0.007

_EMPTY = np.zeros(0, dtype=np.float32)


def _ratio(ratio: float) -> float:
    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"the ratio must be a positive number, not {ratio!r}")
    return value


class TimeStretch:
    """Streaming WSOLA (see the module docstring).

    ``rate`` is the audio's sample rate; ``frame_s`` and ``search_s`` the
    frame length and how far a frame may move (seconds).

    Not thread-safe: one stream, fed and pulled from one thread.
    """

    def __init__(
        self,
        rate: float = 48000.0,
        frame_s: float = FRAME_S,
        search_s: float = SEARCH_S,
    ) -> None:
        self.rate = float(rate)
        #: Output samples per frame step (half a frame).
        self.hop = max(8, int(round(float(frame_s) * self.rate / 2.0)))
        #: Samples per frame.
        self.frame = 2 * self.hop
        #: Samples a frame may move either way.
        self.search = max(1, int(round(float(search_s) * self.rate)))
        k = np.arange(self.frame)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / self.frame)
        self._rise = window[: self.hop].astype(np.float32)
        self._fall = window[self.hop :].astype(np.float32)
        span = self.hop + 2 * self.search
        self._nfft = 1 << int(math.ceil(math.log2(span)))
        self.reset()

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        """Forget the stream: the next block starts a new one, in mono."""
        #: 2 once the stream has had a stereo block, else 1.
        self.channels = 1
        self._buf: np.ndarray = _EMPTY  # input from absolute index _base on
        self._base = 0
        self._pos = 0.0  # where the next frame belongs (absolute, input)
        self._prev: Optional[int] = None  # where the last frame came from
        self._tail: Optional[np.ndarray] = None  # its second half, faded
        self._ready: np.ndarray = _EMPTY  # output made, not yet pulled

    @property
    def lookahead(self) -> int:
        """Input samples a frame may use beyond its place."""
        return self.search + self.frame

    @property
    def _end(self) -> int:
        return self._base + len(self._buf)

    @property
    def pending(self) -> float:
        """Input samples fed but not yet played: from where the next frame
        belongs to the end of the input."""
        return max(0.0, self._end - self._pos)

    @property
    def ready(self) -> int:
        """Output samples made but not yet pulled."""
        return len(self._ready)

    def can_make(self) -> bool:
        """Whether the input holds all a frame may use."""
        return int(math.floor(self._pos)) + self.lookahead <= self._end

    def most_ratio(self, count: int) -> float:
        """The fastest ratio at which :meth:`pull` can give ``count``
        output samples from the input fed so far: ``inf`` when the ratio
        doesn't matter (they are made, or one frame makes them), 0.0 when
        not even one frame can be made."""
        missing = int(count) - len(self._ready)
        if missing <= 0:
            return math.inf
        if not self.can_make():
            return 0.0
        frames = -(-missing // self.hop)
        if frames <= 1:
            return math.inf
        room = self._end - self.lookahead - self._pos - 0.5
        return max(0.0, room / ((frames - 1) * self.hop))

    # -- streaming --------------------------------------------------------

    def feed(self, block: np.ndarray) -> None:
        """Append input: mono (1-D) or stereo (``(n, 2)``)."""
        audio = np.asarray(block, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if audio.ndim == 2 and audio.shape[1] != 2:
            raise ValueError(f"blocks are mono or (n, 2) stereo, not {audio.shape}")
        if audio.ndim not in (1, 2):
            raise ValueError(f"blocks are mono or (n, 2) stereo, not {audio.shape}")
        if not len(audio):
            return
        if audio.ndim == 2 and self.channels == 1:
            self._to_stereo()
        if audio.ndim == 1 and self.channels == 2:
            audio = np.repeat(audio[:, None], 2, axis=1)
        if len(self._buf):
            self._buf = np.concatenate([self._buf, audio])
        else:  # (a copy: the caller's block isn't kept)
            self._buf = np.array(audio, dtype=np.float32, copy=True)

    def _to_stereo(self) -> None:
        self.channels = 2

        def both(x: np.ndarray) -> np.ndarray:
            return np.repeat(x[:, None], 2, axis=1) if x.ndim == 1 else x

        self._buf = both(self._buf) if len(self._buf) else np.zeros((0, 2), np.float32)
        self._ready = both(self._ready) if len(self._ready) else _EMPTY
        if self._tail is not None:
            self._tail = both(self._tail)

    def pull(self, count: int, ratio: float) -> np.ndarray:
        """Up to ``count`` output samples at ``ratio`` input samples per
        output sample (fewer when the input fed so far can't make them)."""
        ratio = _ratio(ratio)
        count = max(0, int(count))
        pieces = [self._ready] if len(self._ready) else []
        have = len(self._ready)
        while have < count and self.can_make():
            made = self._make(ratio)
            pieces.append(made)
            have += len(made)
        if not pieces:
            return self._empty()
        out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        self._ready = out[count:]
        return out[:count]

    def finish(self, ratio: float) -> np.ndarray:
        """The stream ended: the rest of the input as output (at
        ``ratio``), ending in the last frame's fade-out. Then a new stream
        can start."""
        ratio = _ratio(ratio)
        pieces = [self._ready] if len(self._ready) else []
        if len(self._buf):
            end = self._end
            shape = (self.lookahead, 2) if self.channels == 2 else (self.lookahead,)
            self.feed(np.zeros(shape, dtype=np.float32))
            while self._pos < end and self.can_make():
                pieces.append(self._make(ratio))
            if self._tail is not None:
                pieces.append(self._tail)
        out = np.concatenate(pieces) if pieces else self._empty()
        self.reset()
        return out

    def drain(self) -> np.ndarray:
        """The stream goes on unstretched: what was made, then the input
        from where the last frame left it to its end, as it is (the last
        frame's fade-out crossfades into it). The blocks that follow, as
        they are, continue it exactly. Then a new stream can start."""
        pieces = [self._ready] if len(self._ready) else []
        if self._prev is not None and self._tail is not None:
            start = self._prev + self.hop - self._base
            follow = self._buf[start : start + self.hop]
            rise = self._rise if follow.ndim == 1 else self._rise[:, None]
            pieces.append(self._tail[: len(follow)] + rise[: len(follow)] * follow)
            pieces.append(self._buf[start + self.hop :])
        else:
            pieces.append(self._buf[max(0, int(math.floor(self._pos)) - self._base) :])
        pieces = [np.asarray(p, dtype=np.float32) for p in pieces if len(p)]
        out = np.concatenate(pieces) if pieces else self._empty()
        self.reset()
        return out

    def _empty(self) -> np.ndarray:
        return np.zeros((0, 2), np.float32) if self.channels == 2 else _EMPTY

    def _make(self, ratio: float) -> np.ndarray:
        """One frame: a hop of finished output."""
        buf, base, hop = self._buf, self._base, self.hop
        here = int(math.floor(self._pos))
        if self._prev is None:
            at = here  # (the first frame: nothing to line up with)
            frame = buf[at - base : at - base + self.frame]
            out = frame[:hop].copy()  # (at full level from its first sample)
        else:
            at = self._best_place(here)
            frame = buf[at - base : at - base + self.frame]
            rise = self._rise if frame.ndim == 1 else self._rise[:, None]
            assert self._tail is not None
            out = self._tail + rise * frame[:hop]
        fall = self._fall if frame.ndim == 1 else self._fall[:, None]
        self._tail = fall * frame[hop:]
        self._prev = at
        self._pos += ratio * hop
        # Keep what the next frame may use: the half frame that follows the
        # last one (what it lines up with) and its own search.
        keep = min(at + hop, int(math.floor(self._pos)) - self.search)
        if keep > base:
            self._buf = self._buf[keep - base :]
            self._base = keep
        return out

    def _best_place(self, here: int) -> int:
        """Where, within the search of ``here``, the input best continues
        the last frame (its next half frame, as it was in the input)."""
        assert self._prev is not None
        buf, base, hop = self._buf, self._base, self.hop
        low = max(here - self.search, base)
        high = here + self.search
        start = self._prev + hop - base
        template = buf[start : start + hop]
        region = buf[low - base : high - base + hop]
        if template.ndim == 2:
            template = template.sum(axis=1)
            region = region.sum(axis=1)
        template = template.astype(np.float64)
        region = region.astype(np.float64)
        energy_t = float(np.dot(template, template))
        if energy_t <= 1e-12:
            return min(max(here, low), high)  # (silence: its own place)
        count = high - low + 1
        spectrum = np.fft.rfft(region, self._nfft) * np.conj(
            np.fft.rfft(template, self._nfft)
        )
        correlation = np.fft.irfft(spectrum, self._nfft)[:count]
        squares = np.concatenate([[0.0], np.cumsum(region * region)])
        energy = squares[hop : hop + count] - squares[:count]
        score = correlation / np.sqrt(np.maximum(energy, 1e-12 * energy_t))
        return low + int(np.argmax(score))


def time_stretch(
    audio: np.ndarray, ratio: float, rate: float = 48000.0, **kwargs: float
) -> np.ndarray:
    """``audio`` (mono or ``(n, 2)``) played ``ratio`` times as fast, its
    pitch kept: about ``len(audio) / ratio`` samples (a whole signal at
    once; :class:`TimeStretch` streams). ``kwargs``: ``frame_s``,
    ``search_s``."""
    stretch = TimeStretch(rate, **kwargs)
    stretch.feed(audio)
    made = stretch.pull(int(len(audio) / _ratio(ratio)), ratio)
    rest = stretch.finish(ratio)
    return np.concatenate([made, rest]) if len(rest) else made

"""
The speaker's sound while a recording plays faster or slower than real
time (File > Open Recording, the playback strip's Speed).

At 0.5x, 2x or 4x each display frame demodulates the recording's audio at
the recording's own pace (twice the signal per frame at 2x), more or less
than the speaker plays in that time. :class:`PlaybackSound` time-stretches
it (``dsp.time_stretch``: WSOLA, the pitch kept) to the pace the
recording really plays at, and feeds the speaker as a sound card takes
audio:

* **The pace** is measured every frame, squelch open or not
  (:meth:`note`): the signal each frame brings against the time since the
  last frame, over about ``PACE_S``. A frame later than ``STALL_S`` (a
  stall of the window) counts as that long, so a hiccup isn't taken for
  the pace. On a computer that can't keep up the recording plays more
  slowly than chosen (the display loop reads what it manages): the pace
  drops, and the sound follows.
* **The speaker** gets, with each block, what it played since the last
  one: it is kept at its own cushion (``cushion_s``, the AudioSink's
  ``TARGET_S``), or at the longest recent wait between frames when that is
  longer (``HOLD_S``; at most ``MAX_OUT_S``), so frames that come late now
  and then don't make it run dry, and the wait settles back afterwards.
* **The input not yet played** waits in the stretch: what a frame may
  look ahead, plus the largest recent swing between what a block brought
  and what the speaker took meanwhile (a computer that can't keep up
  reads 1x and, once a second, a probe frame at the speed chosen: 0.2 s
  at once at 4x, given out over the second after; a stall's catch-up
  isn't such a pattern). The ratio leans a little slower or faster to
  keep it there (``REFILL_S``, ``DRAIN_S``), drains a pile-up fast
  (``FAST_DRAIN_S``: the pace came back faster than measured), and never
  asks for input that hasn't come (it slows to what there is: the pace
  fell). So there are no gaps and the delay doesn't grow.
* **At 1x** the audio passes through untouched (the same array) while the
  recording keeps up; only while it falls behind (the pace below
  ``ENGAGE_BELOW``) is it stretched to the pace, and once it keeps up
  again the stretch drains and hands back sample for sample.
* :meth:`finish` ends a stream (the squelch closed, the speaker stopped):
  what the stretch still holds plays at once, in at most ``FLUSH_S``.
* **A new stream** (the squelch opened again) starts once the stretch
  holds enough to bring the speaker to its level by itself: what the last
  one left still plays first (the speaker's queue is followed across
  streams, a seek and :meth:`reset`), and no silence is padded in front
  of it, so a squelch that closes for a frame or two doesn't run the
  speaker dry mid-sound. (A speaker opened anew holds less than this
  thinks: AudioSink pads a new stream to its cushion, which covers it.)

Qt-free: the window calls it from its display loop with the time of each
frame (tests drive it with made-up times).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ..dsp.time_stretch import TimeStretch


class PlaybackSound:
    """The time-stretch between a playing recording's audio and the
    speaker (see the module docstring). ``rate``: the audio's rate;
    ``cushion_s``: the audio the speaker keeps queued (AudioSink's
    ``TARGET_S``), which it pads with silence when it runs dry.

    Not thread-safe: the display loop's thread only.
    """

    #: The pace is measured over about this long (seconds).
    PACE_S = 1.5
    #: Frames further apart than this count as this far (a stall).
    STALL_S = 0.25
    #: The most sound one block gives the speaker (seconds).
    MAX_OUT_S = 0.25
    #: Input kept beyond what a frame may look ahead and the largest
    #: recent swing (what the kept input had to give besides a block, or
    #: keep of one for later).
    MARGIN_S = 0.02
    #: The input kept is judged smoothed over about this long; short of
    #: its target it is made up over about REFILL_S, beyond it drained over
    #: about DRAIN_S, and a pile-up beyond the swing once more over
    #: about FAST_DRAIN_S.
    SETTLE_S = 1.0
    REFILL_S = 0.5
    DRAIN_S = 1.5
    FAST_DRAIN_S = 0.5
    #: The largest swing and the longest wait are remembered this long.
    HOLD_S = 3.0
    #: The slowest and fastest the sound may play (input per output).
    MIN_RATIO = 0.25
    MAX_RATIO = 5.0
    #: At 1x the sound is stretched while the pace is below ENGAGE_BELOW,
    #: until it is back at RELEASE_AT.
    ENGAGE_BELOW = 0.97
    RELEASE_AT = 0.99
    #: What is left when a stream ends plays in at most this long.
    FLUSH_S = 0.1

    def __init__(self, rate: float = 48000.0, cushion_s: float = 0.12) -> None:
        self.rate = float(rate)
        self.cushion_s = float(cushion_s)
        self._stretch = TimeStretch(self.rate)
        self._speed = 1.0
        # The speaker, as it plays what it is given in real time: what it
        # held after the last write, and when (kept across streams).
        self._queued = 0.0
        self._spoke_at: Optional[float] = None
        self._waits: Deque[Tuple[float, float]] = deque()
        self._swings: Deque[Tuple[float, float]] = deque()
        self.reset()

    # -- the pace ---------------------------------------------------------

    @property
    def speed(self) -> float:
        """The speed chosen (the strip's Speed)."""
        return self._speed

    def _follow_speed(self, speed: float) -> None:
        """A new speed: the pace is measured anew from it (the stream goes
        on: the ratio moves to it with the next block)."""
        self._speed = float(speed)
        self._a = self._speed * self.PACE_S  # (signal seconds, decaying)
        self._b = self.PACE_S  # (clock seconds, decaying)
        self._noted: Optional[float] = None

    def note(self, seconds: float, now: float, speed: float) -> None:
        """A display frame brought ``seconds`` of the recording's signal
        at monotonic time ``now``, played at ``speed`` (every frame while
        the recording plays, whether or not the speaker plays it)."""
        speed = float(speed)
        if not speed > 0:
            return
        if speed != self._speed:
            self._follow_speed(speed)
        last, self._noted = self._noted, float(now)
        if last is None:
            return
        elapsed = min(max(0.0, float(now) - last), self.STALL_S)
        decay = math.exp(-elapsed / self.PACE_S)
        self._a = self._a * decay + max(0.0, float(seconds))
        self._b = self._b * decay + elapsed

    @property
    def pace(self) -> float:
        """The speed the recording really plays at (x real time)."""
        return self._a / self._b if self._b > 0 else self._speed

    # -- the stream -------------------------------------------------------

    @property
    def stretching(self) -> bool:
        """Whether the sound goes through the stretch now (else as it is)."""
        return self._stretching

    @property
    def ratio(self) -> float:
        """The ratio the last block played at (1.0: as it is)."""
        return self._ratio if self._stretching else 1.0

    @property
    def waiting_s(self) -> float:
        """Seconds of sound the stretch holds, not yet given out."""
        if not self._stretching:
            return 0.0
        stretch = self._stretch
        held = stretch.pending / max(self._ratio, self.MIN_RATIO) + stretch.ready
        return held / self.rate

    def reset(self) -> None:
        """Forget the stream and the pace (a new recording, a seek). What
        the speaker still plays is kept in mind: it is the same speaker."""
        self._follow_speed(self._speed)
        self._end_stream()

    def _end_stream(self) -> None:
        self._stretch.reset()
        self._last: Optional[float] = None  # when this stream's last block came
        self._carry = 0.0  # a fraction of an output sample owed
        self._settled: Optional[float] = None  # input kept, smoothed
        self._waits.clear()
        self._swings.clear()
        self._stretching = False
        self._flowing = False
        self._passing = False  # (blocks played as they are, at 1x)
        self._ratio = 1.0

    def _held_max(
        self, store: Deque[Tuple[float, float]], now: float, value: float
    ) -> float:
        """The largest ``value`` within the last ``HOLD_S`` seconds."""
        while store and store[0][0] < now - self.HOLD_S:
            store.popleft()
        while store and store[-1][1] <= value:
            store.pop()
        store.append((now, value))
        return store[0][1]

    def _speaker_left(self, now: float) -> float:
        """Seconds the speaker still holds at ``now``."""
        if self._spoke_at is None:
            return 0.0
        return max(0.0, self._queued - max(0.0, now - self._spoke_at))

    def _wrote(self, now: float, remaining: float, seconds: float) -> None:
        """The speaker had ``remaining`` seconds at ``now`` and got
        ``seconds`` more (a dry speaker pads a short block to its
        cushion)."""
        if remaining <= 0.0 and 0.0 < seconds < self.cushion_s:
            self._queued = self.cushion_s
        else:
            self._queued = remaining + seconds
        self._spoke_at = now

    def play(self, block: np.ndarray, now: float, last: bool = False) -> np.ndarray:
        """The sound for the speaker for ``block`` (mono or ``(n, 2)``
        audio of the recording, as demodulated) at monotonic time
        ``now``: the block itself at 1x while the recording keeps up, else
        what the speaker plays until the next block, time-stretched.
        ``last``: the stream ends with this block (the squelch closed, the
        speaker stopped): what the stretch still holds comes with it
        (:meth:`finish`)."""
        if not len(block):
            return _joined(block, self.finish(now)) if last else block
        out = self._play(block, float(now))
        return _joined(out, self.finish(now)) if last else out

    def _play(self, block: np.ndarray, now: float) -> np.ndarray:
        seconds = len(block) / self.rate
        pace = self.pace
        one = self._speed == 1.0
        remaining = self._speaker_left(now)
        # (The time since this stream's last block: 0 for its first.)
        elapsed = 0.0 if self._last is None else max(0.0, now - self._last)
        # A block much later than the others (a stall) is no pattern to
        # keep: the speaker ran dry, and plays on as before.
        steady = elapsed <= self.STALL_S
        wait = self._held_max(self._waits, now, elapsed if steady else 0.0)
        self._last = now
        if not self._stretching:
            if one and pace >= self.ENGAGE_BELOW:
                self._passing = True
                self._wrote(now, remaining, seconds)
                return block
            self._stretching = True
            self._stretch.reset()
            self._settled = None
            self._swings.clear()
            # Taking over a sound that plays as it is (at 1x, falling
            # behind): no wait to fill up first. A new sound fills up.
            self._flowing = self._passing and remaining > 0.0
        stretch = self._stretch
        stretch.feed(block)
        lookahead = stretch.lookahead / self.rate
        release = one and pace >= self.RELEASE_AT
        allowance = 0.0 if release or not self._swings else self._swings[0][1]
        target = lookahead + self.MARGIN_S + allowance
        have = stretch.pending / self.rate
        level = max(self.cushion_s, wait)
        if not self._flowing:
            # Starting: once the input kept beyond the target brings the
            # speaker to its level by itself (after what the last stream
            # left; nothing padded in front of it).
            surplus = max(0.0, (have - target) / max(pace, self.MIN_RATIO))
            want = surplus if surplus >= level - remaining else 0.0
        elif remaining > level:
            want = max(0.0, elapsed - 0.1 * (remaining - level))
        else:
            want = min(level - remaining, self.MAX_OUT_S)
        # What the kept input had to give besides this block, or keep of
        # it for later: the pattern to keep enough for. A computer that
        # can't keep up reads 1x and, once a second, a probe frame of the
        # speed chosen: 0.2 s at once at 4x, given out over the second
        # after. (A stall's catch-up isn't a pattern: it drains.)
        if steady and self._flowing:
            swing = abs(seconds - want * pace)
            self._held_max(self._swings, now, swing)
        wanted = want * self.rate + self._carry
        count = int(wanted)
        self._carry = wanted - count
        ratio = self._ratio_for(pace, have - want * pace, target, allowance, elapsed)
        if want <= 0.0:
            ratio = pace  # (nothing to make: what the next block starts at)
        cap = stretch.most_ratio(count)
        ratio = min(max(min(ratio, cap), self.MIN_RATIO), self.MAX_RATIO)
        self._ratio = ratio
        out = stretch.pull(count, ratio)
        if len(out):
            self._flowing = True
        if release and stretch.pending / self.rate <= lookahead + self.MARGIN_S:
            # Keeping up at 1x again: back to the sound as it is.
            drained = stretch.drain()
            out = _joined(out, drained)
            self._stretching = False
        self._wrote(now, remaining, len(out) / self.rate)
        return out

    def _ratio_for(
        self, pace: float, after: float, target: float, allowance: float, elapsed: float
    ) -> float:
        """The pace, leaning toward keeping ``target`` seconds of input
        (``after``: what this block leaves at the pace)."""
        if self._settled is None:
            self._settled = after
        else:
            k = 1.0 - math.exp(-max(elapsed, 1e-3) / self.SETTLE_S)
            self._settled += k * (after - self._settled)
        error = self._settled - target
        if error < 0.0:  # (short of it: slow down a little, soon)
            lean = max(-0.3 * pace, error / self.REFILL_S)
        else:
            lean = min(0.5 * pace, error / self.DRAIN_S)
        ratio = pace + lean
        pile = after - target - allowance
        if pile > 0.0:  # (a pile-up: the pace came back faster)
            ratio += pile / self.FAST_DRAIN_S
        return ratio

    def finish(self, now: Optional[float] = None) -> np.ndarray:
        """The stream ended (the squelch closed, the speaker stopped):
        what the stretch still holds, at once, in at most ``FLUSH_S``
        (faster than the ratio where needed), ending in its fade-out.
        Empty when nothing is held. ``now``: when it goes to the speaker
        (the last block's time when not given)."""
        out = np.zeros(0, dtype=np.float32)
        if self._stretching:
            stretch = self._stretch
            held = stretch.pending / self.rate
            # (Output made but not yet given out plays first.)
            room = max(
                self.FLUSH_S - stretch.ready / self.rate, stretch.hop / self.rate
            )
            ratio = min(max(self._ratio, held / room), self.MAX_RATIO * 2)
            out = stretch.finish(ratio)
        when = self._spoke_at if now is None else float(now)
        if len(out) and when is not None:
            # (The speaker plays it after what it holds.)
            self._wrote(when, self._speaker_left(when), len(out) / self.rate)
        self._end_stream()
        return out


def _joined(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(second):
        return first
    if not len(first):
        return second
    if first.ndim != second.ndim:  # (mono after the stream turned stereo)
        first, second = (
            np.repeat(x[:, None], 2, axis=1) if x.ndim == 1 else x
            for x in (first, second)
        )
    return np.concatenate([first, second])

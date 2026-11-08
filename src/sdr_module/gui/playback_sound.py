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
  ``TARGET_S``), or at the longest recent wait between frames
  (``HOLD_S``, measured every frame, squelch open or not) and a quarter
  more (``WAIT_MARGIN``) when that is longer, so frames that come late
  now and then (a computer that can't keep up: its frames, and the probe
  frame it reads once a second most, take some percent more or less each
  time) don't make it run dry, and the wait settles back afterwards. It
  never holds more than AudioSink's own buffer (``MOST_HELD_S``), a
  stream's end included: what doesn't fit waits in AudioSink for the
  window to hand it over, which a frame that takes long doesn't, and then
  more is queued than this follows (past its ``HIGH_S`` AudioSink drops
  the oldest audio: the start of a sound, or the end of one).
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
  again the stretch drains and hands back sample for sample. A block
  longer than the speaker holds (``MOST_HELD_S``: a recording slower than
  a display block a frame, such as 8 kHz audio in a one-channel file,
  half a second a block) is given out as it plays instead, frame by frame
  (:meth:`more`), its very samples in order: handed over whole it was
  more than AudioSink keeps (past its ``HIGH_S`` it dropped the start of
  the next block, one that came a frame early) and lasted exactly until
  the next one (a frame late, the speaker ran dry). A new sound's first
  piece is shorter than the speaker's cushion, which AudioSink pads in
  front of it with silence: the margin a block a frame late still finds
  to play.
* :meth:`finish` ends a stream (the squelch closed, the speaker stopped):
  what the stretch still holds plays at once, in at most ``FLUSH_S`` and
  in no more than the speaker can still take (faster, or cut short and
  faded, when it holds a long wait).
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
    #: The speaker is kept this much (a fraction) above the longest recent
    #: wait between frames: the next frame may take a little longer (the
    #: probe frames of an overloaded 4x vary by 10-20 %, and with no margin
    #: one of them ran the speaker dry every few seconds).
    WAIT_MARGIN = 0.25
    #: The most sound the speaker holds, a stream's end included (but for
    #: its last fade): AudioSink's own buffer (its BUFFER_S). What doesn't
    #: fit there waits in AudioSink for the window to hand it over between
    #: frames, so a frame that takes long plays only what fits (and then
    #: the speaker holds more than this follows: past AudioSink's HIGH_S it
    #: drops the oldest audio, the start of a sound or the end of one).
    MOST_HELD_S = 0.25
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
        gap = max(0.0, float(now) - last)
        if gap <= self.STALL_S:  # (a stall is no pattern to keep)
            # The longest recent wait between frames, which the speaker is
            # kept above: kept across streams (the squelch closing for a
            # moment doesn't make this computer any faster).
            self._held_max(self._waits, float(now), gap)
        elapsed = min(gap, self.STALL_S)
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
        """Seconds of sound the stretch holds, not yet given out (or at 1x
        what a block longer than the speaker holds has still to give)."""
        if self._held is not None:
            return len(self._held) / self.rate
        if not self._stretching:
            return 0.0
        stretch = self._stretch
        held = stretch.pending / max(self._ratio, self.MIN_RATIO) + stretch.ready
        return held / self.rate

    @property
    def holding(self) -> bool:
        """Whether sound waits here to be given out (the stretch's, or at
        1x the rest of a block longer than the speaker holds): what
        :meth:`finish` ends."""
        return self._stretching or self._held is not None

    def reset(self) -> None:
        """Forget the stream and the pace (a new recording, a seek). What
        the speaker still plays is kept in mind: it is the same speaker."""
        self._follow_speed(self._speed)
        self._end_stream()
        self._waits.clear()

    def _end_stream(self) -> None:
        self._stretch.reset()
        self._last: Optional[float] = None  # when this stream's last block came
        self._carry = 0.0  # a fraction of an output sample owed
        self._settled: Optional[float] = None  # input kept, smoothed
        self._swings.clear()
        self._stretching = False
        self._flowing = False
        self._passing = False  # (blocks played as they are, at 1x)
        # At 1x, the rest of a block longer than the speaker holds, as it
        # is: given out as the speaker plays it (see _give).
        self._held: Optional[np.ndarray] = None
        self._block_s = 0.0  # (the length of this stream's last block)
        self._sounded = False  # (the speaker got some of this stream)
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

    def _longest_wait(self, now: float) -> float:
        """The longest wait between frames within the last ``HOLD_S``."""
        waits = self._waits
        while waits and waits[0][0] < now - self.HOLD_S:
            waits.popleft()
        return waits[0][1] if waits else 0.0

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
        if seconds > 0.0:
            self._sounded = True

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
        self._block_s = len(block) / self.rate
        out = self._play(block, float(now))
        return _joined(out, self.finish(now)) if last else out

    def more(self, now: float) -> np.ndarray:
        """The sound for the speaker at monotonic time ``now`` for a
        display frame that brought no block: a recording slower than a
        display block a frame (2048 samples: a one-channel file of 8 kHz
        audio is 4 kS/s of I/Q, a block every half second at 1x, every
        second at 0.5x) gives its sound out between its blocks too, from
        what the stretch holds, as a block's :meth:`play` would, or at 1x
        from what its last block has still to give, as it is. Empty while
        nothing is held (a recording that keeps up at 1x, its blocks no
        longer than the speaker holds, passes them through as they are) or
        before a sound has started."""
        now = float(now)
        if self._held is not None:
            if self._speed == 1.0:
                # (Whatever the pace measured between blocks: it swings
                # about 1x with each one of a recording this slow.)
                return self._give(now)
            # Another speed chosen between blocks: the stretch takes over
            # with what is held.
            held, self._held = self._held, None
            return self._play(held, now)
        if not (self._stretching and self._flowing):
            return np.zeros(0, dtype=np.float32)
        return self._play(np.zeros(0, dtype=np.float32), now)

    def _level(self, wait: float) -> float:
        """Seconds the speaker is kept at, given the longest recent wait
        between frames (see the module docstring)."""
        most = max(self.cushion_s, self.MOST_HELD_S)
        return min(max(self.cushion_s, wait * (1.0 + self.WAIT_MARGIN)), most)

    def _give(self, now: float) -> np.ndarray:
        """At 1x, from the rest of a block longer than the speaker holds:
        what the speaker played since the last write (it is kept at its
        level, as by the stretch), as it is."""
        held = self._held
        if held is None:
            return np.zeros(0, dtype=np.float32)
        remaining = self._speaker_left(now)
        want = max(0.0, self._level(self._longest_wait(now)) - remaining)
        if remaining <= 0.0:
            if self._sounded and self._spoke_at is not None:
                # The speaker ran dry mid-sound (the window stalled, or a
                # block came later than the margin covers): what it would
                # have played meanwhile is skipped. (Played late instead,
                # each stall made the sound that much later for good: the
                # sound as it is can't catch up as the stretch does.)
                dry = max(0.0, now - self._spoke_at - self._queued)
                held = held[min(len(held), int(round(dry * self.rate))) :]
            # A new sound (or one the speaker ran dry of): its first piece
            # is shorter than the cushion AudioSink pads it to, with silence
            # in front: the margin a block a frame late still finds to play.
            want = min(want, 0.5 * self.cushion_s)
        count = min(len(held), int(round(want * self.rate)))
        out = held[:count]
        self._held = held[count:] if count < len(held) else None
        self._wrote(now, remaining, count / self.rate)
        return out

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
        wait = self._longest_wait(now)
        self._last = now
        # (This block starts the stretch: a new sound, or one that played
        # as it is at 1x taken over.)
        first = not self._stretching
        if not self._stretching:
            if one and pace >= self.ENGAGE_BELOW:
                self._passing = True
                if self._held is None and seconds <= self.MOST_HELD_S:
                    self._wrote(now, remaining, seconds)
                    return block
                # Longer than the speaker holds (or after what is still
                # held): given out as it plays, frame by frame (more).
                audio = np.array(block, dtype=np.float32, copy=True)
                held = self._held
                self._held = audio if held is None else _joined(held, audio)
                return self._give(now)
            if self._held is not None:
                # The stretch takes over: what is held comes first.
                block, self._held = _joined(self._held, block), None
                seconds = len(block) / self.rate
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
        if release and self._block_s > self.MOST_HELD_S:
            # Back at 1x, a recording slower than a display block a frame:
            # what the stretch holds (kept for the frames until the next
            # block) plays on as it is, given out as the speaker plays it
            # (drained fast as a pile-up first, it ran the speaker dry until
            # the next block: 0.15-0.4 s).
            drained = stretch.drain()
            self._held = drained if len(drained) else None
            self._stretching = False
            self._passing = True
            return self._give(now)
        allowance = 0.0 if release or not self._swings else self._swings[0][1]
        target = lookahead + self.MARGIN_S + allowance
        have = stretch.pending / self.rate
        most = max(self.cushion_s, self.MOST_HELD_S)
        level = self._level(wait)
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
        # (Never more than fits AudioSink's own buffer: see MOST_HELD_S.)
        want = min(want, max(0.0, most - remaining))
        starting = first or not self._flowing
        if starting and seconds > self.STALL_S * max(pace, self.MIN_RATIO):
            # A sound's first block holds more than a stall's worth of it:
            # a recording slower than a display block a frame, whose blocks
            # are given out over the frames until the next one (:meth:`more`;
            # 8 kHz audio in a one-channel file: half a second a block).
            # What this one keeps for later is such a swing from the start:
            # taken for a pile-up, it was rushed through, and the speaker
            # ran dry before the next block. (So with the first block the
            # stretch takes over from 1x, another speed chosen.)
            allowance = self._held_max(self._swings, now, seconds - want * pace)
            target = lookahead + self.MARGIN_S + allowance
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
        (faster than the ratio where needed), ending in its fade-out, and
        no more than fits what the speaker holds under ``MOST_HELD_S`` but
        for that fade (cut short and faded when even that is too long).
        At 1x the rest of a block longer than the speaker holds, as it is,
        within the same bounds. Empty when nothing is held. ``now``: when
        it goes to the speaker (the last block's time when not given)."""
        out = np.zeros(0, dtype=np.float32)
        when = self._spoke_at if now is None else float(now)
        if self._held is not None:
            # (Normally a frame or two of it: the next block was due. The
            # rest of a block, when nobody hears it any more, is cut short.)
            hop = self._stretch.hop
            left = 0.0 if when is None else self._speaker_left(when)
            room = max(0.0, self.MOST_HELD_S - left)  # (see MOST_HELD_S)
            fits = min(self.FLUSH_S, room) + hop / self.rate  # (and its fade)
            out = self._held
            most = max(hop, int(fits * self.rate))
            if len(out) > most:
                out = _faded(out[:most], hop)
        elif self._stretching:
            stretch = self._stretch
            held = stretch.pending / self.rate
            hop = stretch.hop / self.rate
            left = 0.0 if when is None else self._speaker_left(when)
            # (What the speaker can still take: see MOST_HELD_S.)
            fits = min(self.FLUSH_S + 2 * hop, max(0.0, self.MOST_HELD_S - left) + hop)
            # (Output made but not yet given out plays first, the last
            # frame's fade-out after it; frames come in whole hops.)
            room = max(fits - 2 * hop - stretch.ready / self.rate, hop)
            ratio = min(max(self._ratio, held / room), self.MAX_RATIO * 4)
            out = stretch.finish(ratio)
            most = max(stretch.hop, int(fits * self.rate))
            if len(out) > most:
                out = _faded(out[:most], stretch.hop)
        if len(out) and when is not None:
            # (The speaker plays it after what it holds.)
            self._wrote(when, self._speaker_left(when), len(out) / self.rate)
        self._end_stream()
        return out


def _faded(audio: np.ndarray, count: int) -> np.ndarray:
    """``audio`` (a copy) faded out over its last ``count`` samples."""
    out = np.array(audio, dtype=np.float32, copy=True)
    n = min(int(count), len(out))
    if n:
        fall = (0.5 + 0.5 * np.cos(np.pi * (np.arange(n) + 1) / n)).astype(np.float32)
        out[-n:] *= fall if out.ndim == 1 else fall[:, None]
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

"""
Streaming Morse code (CW) decoding: a keyed carrier's envelope in, text out.

A CW station keys a carrier on and off: marks of one dit (``.``) and of
three dits (``-``), with gaps of one dit between the elements of a letter,
three between letters and seven between words. Decoding it takes two
steps, each keeping its state from one block to the next, so a stream
decoded 33 ms at a time gives the same text as the whole recording at once:

- :class:`KeyingDetector` decides key down / key up from the envelope. The
  envelope is smoothed over 15 ms and two levels are tracked across
  blocks: the signal's peak (it follows a rising envelope at once and
  decays over seconds) and the noise floor (it follows a falling envelope
  at once and rises over seconds, far more slowly while a strong signal
  keys, so a long carrier is not taken for noise, and faster while nothing
  keys, so a slow swing of the noise level does not open the gate). The
  key goes down when the envelope is 60% of the way from the floor to the
  peak and comes up below 40% (hysteresis), and only while the peak stands
  clear of the floor, so noise alone does not key. A change must hold for
  a few milliseconds (both edges are delayed alike, so the timing is
  kept). Decisions are made at about 1 kHz whatever the input rate; the
  smoothing is vectorised.
- :class:`MorseDecoder` times the marks and gaps (the run in progress is
  carried over block boundaries), learns the dit length from the marks
  (two clusters, dits and dashes, so the speed is found on its own) and
  turns them into letters, words and lines. The first few elements wait
  until the speed is known, so the first letters decode too; a pause of a
  word or more ends a transmission even if it was too short for that. A
  mark much longer than a dash (a tuning carrier) ends the line and is
  left out of the speed; so does a long silence. A lone dit or dash (an E
  or a T as a word of its own: what a crash of static gives) is not shown
  next to words of several letters, nor in a line of nothing else; in
  code spaced letter by letter (Farnsworth practice, a loose hand: a word
  per letter) it shows once the code is heard to go so. A blip of fewer
  than three marks is not shown either. The speed is ``1.2 / dit`` words
  per minute (PARIS), 5 to 60 WPM.

NumPy only; no Qt.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .channelizer import MovingAverage
from .protocols import DecodedMessage, ProtocolType

__all__ = [
    "MORSE_CODE",
    "MorseMessage",
    "KeyingDetector",
    "MorseDecoder",
    "wpm_from_dit",
]

#: Morse code -> character (letters, digits, punctuation and prosigns).
MORSE_CODE: Dict[str, str] = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    ".-.-.-": ".",
    "--..--": ",",
    "..--..": "?",
    ".----.": "'",
    "-.-.--": "!",
    "-..-.": "/",
    "-.--.": "(",
    "-.--.-": ")",
    ".-...": "&",
    "---...": ":",
    "-.-.-.": ";",
    "-...-": "=",
    ".-.-.": "+",
    "-....-": "-",
    "..--.-": "_",
    ".-..-.": '"',
    "...-..-": "$",
    ".--.-.": "@",
    "...---...": "SOS",
    "...-.-": "<SK>",
    "-.-.-": "<KA>",
}

#: Shown for an element sequence that is not a Morse character.
UNKNOWN_CHAR = "*"


def wpm_from_dit(dit_s: float) -> float:
    """Words per minute for a dit of ``dit_s`` seconds (PARIS: 50 dits)."""
    return 1.2 / max(float(dit_s), 1e-6)


@dataclass
class MorseMessage(DecodedMessage):
    """One line of decoded Morse text."""

    text: str = ""
    #: Sending speed when the line ended (words per minute).
    wpm: float = 0.0


def _alpha(rate: float, tau_s: float) -> float:
    """One-pole smoothing coefficient for a time constant at ``rate``."""
    return 1.0 - math.exp(-1.0 / max(rate * tau_s, 1e-9))


def _lone_e_or_t(codes: List[str]) -> bool:
    """Whether the letters of a word (codes such as ``".-"``) are a lone E
    or T: what a crash of static keys."""
    return len(codes) == 1 and len(codes[0]) == 1


class KeyingDetector:
    """Key down / key up from a CW envelope, in blocks of any size.

    ``process(envelope)`` returns one decision per input sample (True: key
    down). The levels and the key state carry over between calls. Below
    ``MIN_LEVEL`` (digital silence) nothing keys.
    """

    #: Decisions per second (the input is smoothed, then read at this rate).
    WORK_RATE = 1000.0
    SMOOTH_S = 0.015
    PEAK_ATTACK_S = 0.005
    PEAK_DECAY_S = 2.0
    FLOOR_FALL_S = 0.05
    FLOOR_RISE_S = 3.0
    #: While the gate is closed (nothing keys) the floor follows a rising
    #: noise level faster, so a slow swing of the band noise (fading, a
    #: receiver's AGC) does not open it; a signal still opens it at once
    #: (the peak follows within milliseconds), and from then on the floor
    #: rises at the usual pace.
    FLOOR_RISE_CLOSED_S = 1.0
    #: While the key is down on a strong signal (peak at least ``STRONG``
    #: times the floor) the floor rises much more slowly, so a long carrier
    #: (a beacon's tuning dash) is not taken for the noise.
    FLOOR_RISE_KEYED_S = 60.0
    STRONG = 4.0
    #: Key down above this fraction of the way from the floor to the peak,
    #: key up below ``RELEASE``.
    PRESS = 0.6
    RELEASE = 0.4
    #: The peak must be this many times the floor before anything keys.
    #: Noise alone, through a +/-250 Hz CW channel, stays below it (no
    #: character in minutes of it), while a keyed carrier 8 dB above the
    #: noise in 500 Hz decodes fully and 6 dB mostly. Once keying, the
    #: ratio may drop to ``GATE_LOW``.
    GATE = 2.6
    GATE_LOW = 2.3
    #: A change of state must hold this long (seconds).
    DEBOUNCE_S = 0.004
    #: The levels settle this long after a reset (the floor starts from
    #: silence and follows the mean level). Meanwhile only a steady
    #: envelope keys: one whose sample-to-sample fluctuation over
    #: ``STEADY_S`` is below ``STEADY_CV`` of its level (a carrier; the
    #: envelope of noise fluctuates by half its level).
    WARMUP_S = 0.3
    STEADY_S = 0.02
    STEADY_CV = 0.3
    MIN_LEVEL = 1e-7

    def __init__(self, rate: float):
        rate = float(rate)
        if not rate > 0:
            raise ValueError(f"rate must be positive, got {rate:g}")
        self.rate = rate
        self._factor = max(1, int(round(rate / self.WORK_RATE)))
        self.work_rate = rate / self._factor
        self._smooth_len = max(1, int(round(self.SMOOTH_S * rate)))
        work = self.work_rate
        self._a_peak_up = _alpha(work, self.PEAK_ATTACK_S)
        self._a_peak_down = _alpha(work, self.PEAK_DECAY_S)
        self._a_floor_down = _alpha(work, self.FLOOR_FALL_S)
        self._a_floor_up = _alpha(work, self.FLOOR_RISE_S)
        self._a_floor_up_closed = _alpha(work, self.FLOOR_RISE_CLOSED_S)
        self._a_floor_up_keyed = _alpha(work, self.FLOOR_RISE_KEYED_S)
        self._debounce = max(1, int(round(self.DEBOUNCE_S * work)))
        self._warmup = int(round(self.WARMUP_S * work))
        self._steady_len = max(2, int(round(self.STEADY_S * rate)))
        self.reset()

    def reset(self) -> None:
        """Forget the stream (levels, key state, smoothing)."""
        self._avg = MovingAverage(self._smooth_len)
        self._phase = 0  # index, in the next block, of the next decision
        self._key = False
        self._pending = 0  # decisions in a row that disagree with the key
        self._peak: Optional[float] = None
        self._floor = 0.0
        self._open = False  # the gate (peak clear of the floor)
        self._settling = self._warmup  # decisions left in the settling time
        # The raw envelope's mean and power over STEADY_S (while settling).
        self._raw_mean = MovingAverage(self._steady_len)
        self._raw_power = MovingAverage(self._steady_len)

    @property
    def key_down(self) -> bool:
        return self._key

    @property
    def peak(self) -> float:
        return float(self._peak or 0.0)

    @property
    def floor(self) -> float:
        return float(self._floor)

    @property
    def signal(self) -> bool:
        """Whether the peak stands clear of the noise floor (a keyed
        signal is there), as of the last decision."""
        return self._open

    def process(self, envelope: np.ndarray) -> np.ndarray:
        """Decisions (bool, True = key down) for each sample of ``envelope``
        (the magnitude of the CW channel; complex input is taken as I/Q)."""
        x = np.abs(np.asarray(envelope)).ravel()
        x = np.nan_to_num(x.astype(np.float64, copy=False))
        n = len(x)
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self._peak is None:
            # Start the smoothing from the first level, not from silence
            # (a floor learned from the warm-up would let noise key), and
            # the steadiness test from "not steady" (a fluctuation as large
            # as the level), which the real samples then replace.
            self._avg.process(np.full(self._smooth_len - 1, x[0]))
            self._raw_mean.process(np.full(self._steady_len - 1, x[0]))
            self._raw_power.process(np.full(self._steady_len - 1, 2.0 * x[0] ** 2))
        smooth = self._avg.process(x)
        picks = np.arange(self._phase, n, self._factor)
        before = self._key
        if len(picks):
            self._phase = int(picks[-1]) + self._factor - n
            steady = None
            if self._settling > 0:
                mean = self._raw_mean.process(x)[picks]
                power = self._raw_power.process(x * x)[picks]
                steady = (mean > self.MIN_LEVEL) & (
                    power - mean * mean < self.STEADY_CV**2 * mean * mean
                )
            decisions = self._track(smooth[picks], steady)
            counts = np.diff(np.append(picks, n))
            out = np.empty(n, dtype=bool)
            out[: picks[0]] = before
            out[picks[0] :] = np.repeat(decisions, counts)
            return out
        self._phase -= n
        return np.full(n, before, dtype=bool)

    def _track(self, values: np.ndarray, steady: Optional[np.ndarray]) -> np.ndarray:
        """The level trackers and the key, one decision per value
        (``steady``: whether the raw envelope is steady there; needed
        while settling)."""
        out = np.empty(len(values), dtype=bool)
        peak = self._peak
        floor = self._floor
        key = self._key
        pending = self._pending
        gate_open = self._open
        a_pu, a_pd = self._a_peak_up, self._a_peak_down
        a_fd, a_fu = self._a_floor_down, self._a_floor_up
        a_fk = self._a_floor_up_keyed
        a_fc = self._a_floor_up_closed
        strong_ratio = self.STRONG
        gate_high, gate_low = self.GATE, self.GATE_LOW
        press, release = self.PRESS, self.RELEASE
        debounce = self._debounce
        min_level = self.MIN_LEVEL
        settling = self._settling
        for i, v in enumerate(values.tolist()):
            if peak is None:
                peak, floor = v, 0.0  # before the stream: silence
            peak += (a_pu if v > peak else a_pd) * (v - peak)
            if settling > 0:
                # Settling: while the key is up the floor follows the mean
                # level (it then sinks to its usual place, low in the noise,
                # rather than rising to it). Only a steady envelope keys.
                settling -= 1
                if not key:
                    floor += a_fd * (v - floor)
                gate_open = (
                    steady is not None and bool(steady[i]) and peak > gate_high * floor
                )
            else:
                if v < floor:
                    floor += a_fd * (v - floor)
                elif key and peak > strong_ratio * floor:
                    floor += a_fk * (v - floor)
                else:
                    floor += (a_fu if gate_open else a_fc) * (v - floor)
                if peak < min_level:
                    gate_open = False
                elif gate_open:
                    gate_open = peak > gate_low * floor
                else:
                    gate_open = peak > gate_high * floor
            span = peak - floor
            if key:
                want = gate_open and v >= floor + release * span
            else:
                want = gate_open and v > floor + press * span
            if want != key:
                pending += 1
                if pending >= debounce:
                    key = want
                    pending = 0
            else:
                pending = 0
            out[i] = key
        self._peak = peak
        self._floor = float(floor)
        self._key = key
        self._pending = pending
        self._open = gate_open
        self._settling = settling
        return out


class MorseDecoder:
    """Text from key up / key down decisions, in blocks of any size.

    ``process(keying)`` takes the decisions (one per sample at ``rate``;
    anything above 0.5 is key down) and returns the lines completed in the
    block as :class:`MorseMessage` objects. :attr:`text` holds everything
    decoded so far (words separated by spaces, lines by newlines),
    :attr:`line` the line still being received and :attr:`wpm` the speed.
    """

    #: Gaps (in dits) that end a letter and a word.
    LETTER_GAP = 2.0
    WORD_GAP = 5.0
    #: Marks shorter than this (in dits) are noise, not elements.
    GLITCH = 0.25
    #: A mark longer than this (in dits, and ``CARRIER_MIN_S``) is a
    #: carrier, not an element: it ends the line. Until the speed is
    #: known only a mark longer than that at the slowest speed is surely
    #: one (a dash at 5 WPM is 0.72 s); the speed is measured from the
    #: others, leaving out those that are carriers at the speed they give.
    CARRIER_DITS = 6.0
    CARRIER_MIN_S = 0.4
    #: Silence that ends a line: this long (seconds) and ``LINE_GAP_DITS``.
    LINE_GAP_S = 2.5
    LINE_GAP_DITS = 10.0
    #: Marks remembered for the speed estimate; how many it takes first.
    MARKS_KEPT = 16
    CALIBRATE_MARKS = 6
    #: A gap this long (seconds), or ``END_GAP_DITS`` at the speed the
    #: marks heard so far suggest, ends a transmission: what was heard is
    #: decoded even if it had too few marks for a confident speed, when it
    #: had at least ``CALIBRATE_GAP_MARKS`` marks (fewer are a blip of
    #: noise and are dropped, after ``CALIBRATE_GAP_S``).
    CALIBRATE_GAP_S = 1.5
    END_GAP_DITS = 7.0
    CALIBRATE_GAP_MARKS = 3
    WPM_RANGE = (5.0, 60.0)
    #: Before the speed is known, marks all of one kind are dashes when a
    #: gap between them is shorter than this many times their length (the
    #: elements of a letter are a dit apart), else dits.
    DASH_GAP = 0.6
    #: A word that is a lone E or T (what a crash of static keys) waits.
    #: Next to a word of several letters it is static: dropped.
    #: Next to a word of one letter, up to this many in a row show while
    #: the code reads letter by letter (Farnsworth practice, a loose hand):
    #: two of the last ``SPACED_WORDS`` other words were one letter, one of
    #: them of three or more elements (static keys dits, dashes and now and
    #: then an I, A, N or M; a "K" ends many messages).
    HELD_WORDS = 4
    SPACED_WORDS = 4
    #: A lone dit shorter than this (in dits) is a blip of noise, not an E.
    SOLID_DIT = 0.6

    def __init__(self, rate: float, wpm: float = 20.0, line_chars: int = 60):
        rate = float(rate)
        if not rate > 0:
            raise ValueError(f"rate must be positive, got {rate:g}")
        self.rate = rate
        self.default_wpm = float(wpm)
        self.line_chars = max(8, int(line_chars))
        self.reset()

    def reset(self) -> None:
        """Forget the stream, the speed and the text."""
        self._dit = self.rate * 1.2 / self.default_wpm  # samples
        self._calibrated = False
        self._key = False
        self._run = 0  # samples in the current state
        self._gap = 0  # the gap before the mark in progress
        self._stage = 3  # gap stages done (0 none ... 3 line ended)
        self._marks: Deque[float] = deque(maxlen=self.MARKS_KEPT)
        self._queue: List[Tuple[str, int]] = []  # events waiting for a speed
        self._replaying = False
        self._letter = ""
        self._mark_dits = 1.0  # the last mark (dits)
        self._word = ""  # letters of the word in progress not shown yet
        self._codes: List[str] = []  # their codes
        self._word_dits = 1.0  # its last dit or dash (dits)
        self._word_shown = False  # its letters show as they come
        self._held: List[str] = []  # lone Es and Ts waiting (HELD_WORDS)
        self._after_one = False  # the line's last word was one letter
        # The last SPACED_WORDS other words: (one letter, of 3+ elements).
        self._recent: Deque[Tuple[bool, bool]] = deque(maxlen=self.SPACED_WORDS)
        self._pending_space = False
        self.line = ""
        self.text = ""
        self.characters = 0
        self._elapsed = 0  # samples processed
        self._messages: List[MorseMessage] = []

    # -- speed ---------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        """Whether the speed has been measured (decoding has started)."""
        return self._calibrated

    @property
    def dit_s(self) -> float:
        return self._dit / self.rate

    @property
    def wpm(self) -> float:
        """The speed in words per minute (the default until measured)."""
        return wpm_from_dit(self.dit_s)

    def _clamp_dit(self, dit: float) -> float:
        low, high = self.WPM_RANGE
        return min(max(dit, self.rate * 1.2 / high), self.rate * 1.2 / low)

    @property
    def _longest_element(self) -> float:
        """Samples in the longest mark that may be an element at all (a
        carrier at the slowest speed is longer)."""
        return self.CARRIER_DITS * self.rate * 1.2 / self.WPM_RANGE[0]

    def _estimate(self) -> Optional[Tuple[float, bool]]:
        """``(dit in samples, dits and dashes both seen)`` from the recent
        marks, or None without any. The longest marks are left out while
        they are carriers (a tuning carrier among the first letters): at
        the speed the other marks give, when those show dits and dashes,
        or at the speed of them all."""
        marks = sorted(self._marks)
        if not marks:
            return None
        estimate = self._clusters(marks)
        while len(marks) > 1:
            rest = self._clusters(marks[:-1])
            if not (
                self._is_carrier(marks[-1], estimate[0])
                or (rest[1] and self._is_carrier(marks[-1], rest[0]))
            ):
                break
            marks.pop()
            estimate = rest
        return estimate

    def _clusters(self, marks: List[float]) -> Tuple[float, bool]:
        """``(dit in samples, dits and dashes both seen)`` from ``marks``
        (sorted, at least one)."""
        count = len(marks)
        if count >= 2:
            logs = np.log(np.asarray(marks, dtype=np.float64))
            best_split, best_score = 0, -1.0
            for split in range(1, count):
                a, b = logs[:split], logs[split:]
                score = split * (count - split) * (b.mean() - a.mean()) ** 2
                if score > best_score:
                    best_split, best_score = split, score
            short = marks[:best_split]
            long = marks[best_split:]
            short_mean = sum(short) / len(short)
            long_mean = sum(long) / len(long)
            if long_mean >= 2.0 * short_mean:
                dit = (sum(short) + sum(long) / 3.0) / count
                return self._clamp_dit(dit), True
        mean = sum(marks) / count
        return self._clamp_dit(self._one_kind_dit(mean)), False

    def _one_kind_dit(self, mean: float) -> float:
        """The dit when the marks (``mean`` samples long) are all of one
        kind. Until the speed is known the gaps between them tell: marks
        about three times the shortest gap are dashes (a letter's elements
        are a dit apart), marks as long as it (S, H, 5) or shorter
        (letters of one element) dits, unless longer than a dit at the
        slowest speed: then they are dashes, letters apart (T T T).
        Otherwise (no gap between marks yet, or the speed is known) they
        are dits if nearer one dit than three: slow dits would be dashes at
        the default speed."""
        if not self._calibrated:
            # The queued gaps after the first mark (the one before it is
            # the silence before the transmission).
            gaps = [length for kind, length in self._queue[2:] if kind == "gap"]
            if gaps:
                slowest_dit = self.rate * 1.2 / self.WPM_RANGE[0]
                if min(gaps) < self.DASH_GAP * mean or mean > slowest_dit:
                    return mean / 3.0
                return mean
        return mean if mean < 2.0 * self._dit else mean / 3.0

    def _calibrate(self) -> None:
        """Measure the speed from the marks heard so far and decode the
        elements that waited for it."""
        estimate = self._estimate()
        if estimate is not None:
            self._dit = estimate[0]
        self._calibrated = True
        queued, self._queue = self._queue, []
        self._replaying = True
        try:
            for kind, length in queued:
                if kind == "mark":
                    self._mark(length)
                else:
                    self._stage = 0
                    self._gap_reached(length)
        finally:
            self._replaying = False
        self._stage = 0  # the gap after the last queued mark starts afresh

    # -- events --------------------------------------------------------------

    def _mark_ended(self, length: int) -> None:
        """A key-down run of ``length`` samples ended."""
        self._key = False
        if length < self.GLITCH * self._dit:
            # Noise, not an element: the gap before it goes on.
            self._run = self._gap + length
            return
        self._run = 0
        if self._calibrated:
            self._mark(length)
        else:
            self._queue.append(("gap", self._gap))
            self._queue.append(("mark", length))
            if length > self._longest_element:
                # A carrier, longer than an element at any speed: what was
                # heard before it is a transmission of its own.
                self._end_transmission()
            else:
                self._marks.append(float(length))
                estimate = self._estimate()
                if len(self._marks) >= 2 * self.CALIBRATE_MARKS or (
                    len(self._marks) >= self.CALIBRATE_MARKS
                    and estimate is not None
                    and estimate[1]
                ):
                    self._calibrate()
        self._stage = 0

    def _is_carrier(self, length: float, dit: Optional[float] = None) -> bool:
        """Whether a mark is a carrier at the speed of ``dit`` (samples;
        the measured one by default)."""
        return (
            length > self.CARRIER_DITS * (self._dit if dit is None else dit)
            and length > self.CARRIER_MIN_S * self.rate
        )

    def _mark(self, length: int) -> None:
        """Classify a mark (the speed is known). It counts in the speed
        first, so a speed measured wrong (the first marks were all dits,
        taken for dashes) is put right by the next dash rather than taking
        that dash for a carrier."""
        if not self._replaying and length <= self._longest_element:
            self._marks.append(float(length))
            estimate = self._estimate()
            if estimate is not None:
                self._dit = estimate[0]
        if self._is_carrier(length):
            self._end_letter()
            self._end_line()
            return
        self._mark_dits = length / self._dit
        self._letter += "." if length < 2.0 * self._dit else "-"
        if len(self._letter) > 9:  # not Morse: give up on this letter
            self._end_letter()

    def _gap_reached(self, length: int) -> None:
        """The gap in progress is ``length`` samples long: do what that
        means and has not been done yet (end the letter, the word, the
        line). Calling it again with a longer length does the rest, so it
        makes no difference where the blocks end."""
        if not self._calibrated:
            if not (self._queue and self._transmission_over(length)):
                return
            # The end of a transmission: decode what it had.
            if not self._end_transmission():
                return
        dit = self._dit
        if self._stage < 1 and length >= self.LETTER_GAP * dit:
            self._stage = 1
            self._end_letter()
        if self._stage < 2 and length >= self.WORD_GAP * dit:
            self._stage = 2
            self._space()
        if (
            self._stage < 3
            and length >= self.LINE_GAP_S * self.rate
            and length >= self.LINE_GAP_DITS * dit
        ):
            self._stage = 3
            self._end_line()

    def _transmission_over(self, gap: int) -> bool:
        """Whether a gap of ``gap`` samples ends the queued transmission
        (the speed is not known yet)."""
        if gap >= self.CALIBRATE_GAP_S * self.rate:
            return True
        if len(self._marks) < self.CALIBRATE_GAP_MARKS:
            return False
        estimate = self._estimate()
        return estimate is not None and gap >= self.END_GAP_DITS * estimate[0]

    def _end_transmission(self) -> bool:
        """The queued transmission ended before the speed was known: decode
        it if it had enough marks to be one, else drop it (a blip of
        noise). Whether it was decoded."""
        if len(self._marks) >= self.CALIBRATE_GAP_MARKS:
            self._calibrate()
            return True
        self._queue = []
        self._marks.clear()
        return False

    # -- text ----------------------------------------------------------------

    def _emit(self, chars: str) -> None:
        if self._pending_space:
            self._pending_space = False
            self.line += " "
            self.text += " "
        self.line += chars
        self.text += chars

    def _end_letter(self) -> None:
        if not self._letter:
            return
        code, self._letter = self._letter, ""
        char = MORSE_CODE.get(code, UNKNOWN_CHAR)
        self._codes.append(code)
        if self._word_shown:
            self._show(char)
            return
        self._word += char
        if len(code) == 1:
            self._word_dits = self._mark_dits
        if _lone_e_or_t(self._codes) or (len(self._codes) == 1 and self._held):
            # It waits: maybe static (see HELD_WORDS), or a word of one
            # letter that decides about the words that waited before it.
            return
        # A word of several letters: the words that waited were static.
        self._held = []
        self._show_word()

    def _show_word(self) -> None:
        """Show the word in progress so far; its next letters show as they
        come."""
        if self.line:
            self._pending_space = True
        for char in self._word:
            self._show(char)
        self._word = ""
        self._word_shown = True

    def _end_word(self) -> None:
        """The word in progress ended (a word gap, or the end of the line)."""
        self._end_letter()
        codes, self._codes = self._codes, []
        if not codes:
            return
        if not self._word_shown and _lone_e_or_t(codes):
            # It waits (a lone dit much shorter than a dit is a blip).
            if not (self._word == "E" and self._word_dits < self.SOLID_DIT):
                self._held.append(self._word)
            self._word = ""
            return
        one = len(codes) == 1
        self._recent.append((one, one and len(codes[0]) >= 3))
        if not self._word_shown:
            # A word of one letter after words that waited: they show
            # with it if the code reads letter by letter.
            self._show_held(self._letter_by_letter())
            self._show_word()
        self._word_shown = False
        self._after_one = one

    def _letter_by_letter(self) -> bool:
        """Whether the code reads letter by letter (see HELD_WORDS)."""
        ones = [longer for one, longer in self._recent if one]
        return len(ones) >= 2 and any(ones)

    def _show_held(self, show: bool) -> None:
        """Show the words that waited (``show``: up to ``HELD_WORDS`` of
        them), or drop them."""
        held, self._held = self._held, []
        if not show or len(held) > self.HELD_WORDS:
            return
        for word in held:
            if self.line:
                self._pending_space = True
            for letter in word:
                self._show(letter)

    def _show(self, char: str) -> None:
        self.characters += 1
        self._emit(char)

    def _space(self) -> None:
        """A word gap: the next character starts a new word (the space is
        added with it, so :attr:`text` only ever grows)."""
        self._end_word()
        if self.line and not self._pending_space:
            if len(self.line) >= self.line_chars:
                self._end_line()
            else:
                self._pending_space = True

    def _end_line(self) -> None:
        self._end_word()
        # Words that waited after a word of one letter show if the code
        # reads letter by letter; else (a line of nothing else, too) they
        # are what crashes of static key.
        self._show_held(
            self._after_one and bool(self.line) and self._letter_by_letter()
        )
        self._after_one = False
        self._pending_space = False
        line = self.line.strip()
        self.line = ""
        if not line:
            return
        self.text += "\n"
        self._messages.append(
            MorseMessage(
                protocol=ProtocolType.CW,
                timestamp=self._elapsed / self.rate,
                raw_bits=b"",
                valid=True,
                text=line,
                wpm=round(self.wpm, 1),
            )
        )

    def set_wpm(self, wpm: float) -> None:
        """Expect ``wpm`` words per minute (the speed is still measured)."""
        self.default_wpm = float(wpm)
        if not self._calibrated:
            self._dit = self._clamp_dit(self.rate * 1.2 / self.default_wpm)

    def flush(self) -> List[MorseMessage]:
        """End the line in progress (e.g. at the end of a recording),
        with the mark under way if the key is down."""
        if self._key:
            self._mark_ended(self._run)
        if not self._calibrated and self._queue:
            self._end_transmission()
        self._end_line()
        messages, self._messages = self._messages, []
        return messages

    # -- stream --------------------------------------------------------------

    def process(self, keying: np.ndarray) -> List[MorseMessage]:
        """Decode a block of key decisions; returns the lines completed."""
        k = np.asarray(keying).ravel() > 0.5
        n = len(k)
        if n:
            self._elapsed += n
            edges = np.flatnonzero(k[1:] != k[:-1]) + 1
            starts = np.concatenate(([0], edges))
            ends = np.append(edges, n)
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                state = bool(k[start])
                if state != self._key:
                    self._switch(state)
                self._run += end - start
            if not self._key:
                self._gap_reached(self._run)
        messages, self._messages = self._messages, []
        return messages

    def _switch(self, state: bool) -> None:
        """The key changed to ``state``; the run before it is complete."""
        if state:  # a gap ended, a mark starts
            self._gap_reached(self._run)
            self._gap = self._run
            self._key = True
            self._run = 0
        else:  # a mark ended
            self._mark_ended(self._run)

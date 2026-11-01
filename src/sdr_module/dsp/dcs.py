"""
DCS: digital-coded squelch (Motorola's DPL, CDCSS), the digital cousin of
the CTCSS tone squelch.

A DCS transmitter sends a 23-bit codeword over and over at 134.4 bit/s,
NRZ straight onto the FM carrier under the voice (band-limited to about
300 Hz, at roughly the deviation a CTCSS tone gets). The codeword is a
Golay (23,12) code word: 12 data bits, the code's three octal digits (9
bits, least significant first) followed by a fixed 0, 0, 1, then 11 check
bits that make the word a multiple of the generator polynomial
``x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1`` (:data:`DCS_GENERATOR`). It is
sent least significant bit first; a 1 is a positive deviation in normal
polarity, a negative one in inverted polarity ("D023I"), which sends every
bit inverted. Because the code is cyclic, every inverted code sends the
same bits as some normal one, shifted (023 inverted is 047 normal); a
receiver can't tell those apart, and doesn't need to.

* :data:`DCS_CODES`: the 104 standard codes (CHIRP's list; written as
  their octal digits, 23 for "023").
* :func:`dcs_codeword`, :func:`dcs_waveform`: the encoder (tests and the
  demo device).
* :func:`format_dcs` / :func:`parse_dcs` (``"D023N"``), :func:`dcs_label`
  (``"DCS 023 N"``), :func:`dcs_same_signal`.
* :class:`DcsDetector`: the code heard on FM discriminator audio, a block
  at a time.

Plain NumPy, no Qt.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .filters import _kaiser_lowpass

#: The 104 standard DCS codes, written as their octal digits (23 = "023").
DCS_CODES: Tuple[int, ...] = (
    23, 25, 26, 31, 32, 36, 43, 47, 51, 53, 54, 65, 71, 72, 73, 74,
    114, 115, 116, 122, 125, 131, 132, 134, 143, 145, 152, 155, 156, 162,
    165, 172, 174, 205, 212, 223, 225, 226, 243, 244, 245, 246, 251, 252,
    255, 261, 263, 265, 266, 271, 274, 306, 311, 315, 325, 331, 332, 343,
    346, 351, 356, 364, 365, 371, 411, 412, 413, 423, 431, 432, 445, 446,
    452, 454, 455, 462, 464, 465, 466, 503, 506, 516, 523, 526, 532, 546,
    565, 606, 612, 624, 627, 631, 632, 654, 662, 664, 703, 712, 723, 731,
    732, 734, 743, 754,
)  # fmt: skip
#: Bit rate (bit/s).
DCS_BIT_RATE = 134.4
#: The Golay (23,12) generator polynomial, bit i for x^i.
DCS_GENERATOR = 0xC75
_BITS = 23
_FULL = (1 << _BITS) - 1

Code = Tuple[int, bool]  # (code, inverted)


def _octal_value(code: int) -> int:
    """The 9-bit value of a code written as octal digits (23 -> 19)."""
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
        raise TypeError(f"a DCS code is an integer such as 23, not {code!r}")
    text = f"{int(code):03d}"
    if len(text) != 3 or not set(text) <= set("01234567"):
        raise ValueError(f"not a DCS code (three octal digits): {code!r}")
    return int(text, 8)


def _remainder(word: int, poly: int) -> int:
    degree = poly.bit_length() - 1
    while word.bit_length() - 1 >= degree:
        word ^= poly << (word.bit_length() - 1 - degree)
    return word


def dcs_codeword(code: int, inverted: bool = False) -> int:
    """The 23-bit word ``code`` sends, bit 0 sent first (inverted: every
    bit inverted). Any three-octal-digit code is accepted."""
    data = _octal_value(code) | (0b100 << 9)
    check = _remainder(data << 11, DCS_GENERATOR)
    word = data | (check << 12)
    return word ^ _FULL if inverted else word


def _rotations(word: int) -> List[int]:
    return [((word >> s) | (word << (_BITS - s))) & _FULL for s in range(_BITS)]


def _build_tables() -> Tuple[Dict[int, Code], Dict[Code, Code]]:
    """Every shift of every standard code's word (normal codes first, so
    a word both a normal and an inverted code send is named as the normal
    one), and each code's twin of the other polarity."""
    heard: Dict[int, Code] = {}
    by_shape: Dict[Tuple[int, bool], Code] = {}  # (smallest shift, polarity)
    for inverted in (False, True):
        for code in DCS_CODES:
            shifts = _rotations(dcs_codeword(code, inverted))
            for word in shifts:
                heard.setdefault(word, (code, inverted))
            by_shape[(min(shifts), inverted)] = (code, inverted)
    twins: Dict[Code, Code] = {}
    for (shape, inverted), name in by_shape.items():
        twin = by_shape.get((shape, not inverted))
        if twin is not None:
            twins[name] = twin
    return heard, twins


_HEARD, _TWINS = _build_tables()


def dcs_same_signal(code: int, inverted: bool = False) -> Optional[Code]:
    """The standard code of the other polarity that sends the same bits
    (023 normal and 047 inverted), or None."""
    return _TWINS.get((int(code), bool(inverted)))


def format_dcs(code: int, inverted: bool = False) -> str:
    """``"D023N"`` / ``"D023I"``: how radios and settings write a code."""
    return f"D{int(code):03d}{'I' if inverted else 'N'}"


def dcs_label(code: int, inverted: bool = False) -> str:
    """``"DCS 023 N"``: a code as the application shows it."""
    return f"DCS {int(code):03d} {'I' if inverted else 'N'}"


def parse_dcs(text: object) -> Optional[Code]:
    """``(code, inverted)`` of a standard code written ``"D023N"``,
    ``"023I"``, ``"23"``, ``"DCS 023 N"`` (R, for reverse, is inverted
    too); None for anything else."""
    if isinstance(text, bool) or not isinstance(text, (str, int)):
        return None
    s = str(text).strip().upper().replace(" ", "")
    if s.startswith("DCS"):
        s = s[3:]
    elif s.startswith("D"):
        s = s[1:]
    inverted = False
    if s[-1:] in ("N", "I", "R"):
        inverted = s[-1] != "N"
        s = s[:-1]
    if not s or len(s) > 3 or not s.isdigit():
        return None
    code = int(s)
    return (code, inverted) if code in DCS_CODES else None


def dcs_waveform(
    code: int,
    inverted: bool = False,
    rate: float = 48000.0,
    seconds: Optional[float] = None,
    samples: Optional[int] = None,
    start: int = 0,
    bit_rate: float = DCS_BIT_RATE,
    smooth: bool = True,
) -> np.ndarray:
    """The DCS signal of ``code`` as +1 (a 1 in normal polarity) and -1
    levels at ``rate``: ``samples`` (or ``seconds``) of it from sample
    ``start`` of the endless stream.

    ``smooth`` band-limits it as transmitters do (flat to 250 Hz, nothing
    above 350 Hz); taken from the endless stream on both sides, so pieces
    and whole-word loops join seamlessly. Otherwise the plain NRZ levels.
    """
    rate = float(rate)
    count = (
        int(samples)
        if samples is not None
        else int(round(float(seconds or 0.0) * rate))
    )
    word = dcs_codeword(code, inverted)
    pad = int(math.ceil(0.08 * rate)) if smooth else 0
    index = np.arange(int(start) - pad, int(start) + count + pad, dtype=np.float64)
    bit = np.floor(index * float(bit_rate) / rate).astype(np.int64) % _BITS
    levels = np.where((word >> bit) & 1, 1.0, -1.0)
    if not smooth or levels.size == 0:
        return levels[pad : pad + count]
    spectrum = np.fft.rfft(levels)
    f = np.fft.rfftfreq(levels.size, 1.0 / rate)
    taper = np.clip((350.0 - f) / 100.0, 0.0, 1.0)
    response = 0.5 - 0.5 * np.cos(np.pi * taper)
    smoothed = np.fft.irfft(spectrum * response, levels.size)
    return smoothed[pad : pad + count]


class DcsDetector:
    """The DCS code heard on FM audio, a block at a time.

    The input is the discriminator's audio *before* any DC blocking (a
    high-pass droops the long runs of equal bits), at ``sample_rate``.
    It is low-passed to the DCS band (flat to 250 Hz, voice above 300 Hz
    out) at about 2.4 kHz. Every ``UPDATE_S`` the last two or three words
    are read: the level halfway between the bits' two levels (a tuning
    offset moves it), the bit timing from where the signal crosses that
    level (crossings fall on bit boundaries: they must line up to at
    least ``MIN_ALIGN``), each bit from the middle half of its time, and
    the words added up bit by bit (the code repeats every 23 bits) before
    deciding. The words must agree (at most ``MAX_ERRORS`` disagreements
    a word); the 23 bits read are then some shift of the codeword, looked
    up (one wrong bit forgiven) among every standard code's shifts.

    A code stays reported for ``HOLD_S`` after it was last read. It takes
    about 0.4 s to recognise one. The answers don't depend on how the
    audio is cut into blocks.

    :attr:`detected` is the code heard (as the normal code where a normal
    and an inverted one send the same bits), :meth:`is_present` asks about
    one code (the squelch).
    """

    UPDATE_S = 0.05
    HOLD_S = 0.25
    MIN_ALIGN = 0.6
    MAX_ERRORS = 2.0
    _TARGET_RATE = 2400.0

    def __init__(self, sample_rate: float) -> None:
        self._rate = float(sample_rate)
        self._factor = max(1, int(round(self._rate / self._TARGET_RATE)))
        self._out_rate = self._rate / self._factor
        self._spb = self._out_rate / DCS_BIT_RATE
        stop = max(self._out_rate - 300.0, 400.0)
        # (Reversed: applied as dot products with the newest sample last.)
        self._taps1 = _kaiser_lowpass(300.0, stop, self._rate)[::-1].copy()
        self._taps2 = _kaiser_lowpass(250.0, 400.0, self._out_rate)[::-1].copy()
        self._keep = int(math.ceil((3 * _BITS + 2) * self._spb))
        self._update_n = max(1, int(round(self.UPDATE_S * self._out_rate)))
        self._hold_n = int(round(self.HOLD_S * self._out_rate))
        self._probe = np.array([-0.25, -0.125, 0.0, 0.125, 0.25]) * self._spb
        self.reset()

    def reset(self) -> None:
        """Forget the audio (a new stream)."""
        self._history1 = np.zeros(len(self._taps1) - 1)
        self._position = 0
        self._history2 = np.zeros(len(self._taps2) - 1)
        self._buffer = np.zeros(self._keep)
        self._filled = 0
        self._since = 0
        self._word: Optional[int] = None  # the last 23 bits read reliably
        self._age = 0  # decimated samples since then
        self._detected: Optional[Code] = None

    # -- answers -------------------------------------------------------- #
    @property
    def detected(self) -> Optional[Code]:
        """``(code, inverted)`` heard lately, or None."""
        return self._detected

    def is_present(self, code: int, inverted: bool = False) -> bool:
        """Whether the bits heard lately are ``code``'s (with ``inverted``
        polarity), one wrong bit forgiven."""
        word = self._word
        if word is None:
            return False
        try:
            target = dcs_codeword(code, inverted)
        except (TypeError, ValueError):
            return False
        return any(bin(word ^ r).count("1") <= 1 for r in _rotations(target))

    # -- work ----------------------------------------------------------- #
    def process(self, audio: np.ndarray) -> Optional[Code]:
        """Feed a block of discriminator audio; returns :attr:`detected`."""
        x = np.asarray(audio, dtype=np.float64).ravel()
        if x.size == 0:
            return self._detected
        y = self._decimate(x)
        if y.size:
            buf = np.concatenate((self._history2, y))
            self._history2 = buf[y.size :].copy()
            z = (
                np.lib.stride_tricks.sliding_window_view(buf, len(self._taps2))
                @ self._taps2
            )
            self._take(z)
        return self._detected

    def _decimate(self, x: np.ndarray) -> np.ndarray:
        """Low-pass and keep every ``factor``-th sample (streaming)."""
        length = len(self._taps1)
        buf = np.concatenate((self._history1, x))
        self._history1 = buf[len(buf) - (length - 1) :].copy()
        start = self._position
        self._position += len(x)
        first = -(-start // self._factor) * self._factor  # the next kept sample
        rows = np.arange(first - start, len(x), self._factor)
        if rows.size == 0:
            return np.zeros(0)
        windows = np.lib.stride_tricks.sliding_window_view(buf, length)[rows]
        return np.asarray(windows @ self._taps1, dtype=np.float64)

    def _take(self, z: np.ndarray) -> None:
        """Buffer DCS-band samples, reading the words every ``_update_n``."""
        pos = 0
        while pos < z.size:
            room = self._update_n - self._since
            piece = z[pos : pos + room]
            pos += piece.size
            self._buffer = np.concatenate((self._buffer[piece.size :], piece))
            self._filled = min(self._keep, self._filled + piece.size)
            self._since += piece.size
            self._age += piece.size
            if self._since >= self._update_n:
                self._since = 0
                self._read()

    def _read(self) -> None:
        found: Optional[Code] = None
        word: Optional[int] = None
        for words in (3, 2):
            length = int(math.ceil((words * _BITS + 1) * self._spb))
            if self._filled >= length:
                word = self._read_word(self._buffer[-length:], words)
                # (Bits that are no standard code, e.g. the noise as a
                # carrier drops, are no reading.)
                found = None if word is None else self._name(word)
                if found is not None:
                    break
        if found is not None:
            self._word, self._age = word, 0
            self._detected = found
        elif self._age > self._hold_n:
            self._word = None
            self._detected = None

    @staticmethod
    def _name(word: int) -> Optional[Code]:
        """The standard code ``word`` is a shift of (one wrong bit
        forgiven: codewords are 7 bits apart, so at most one fits)."""
        found = _HEARD.get(word)
        if found is not None:
            return found
        for i in range(_BITS):
            found = _HEARD.get(word ^ (1 << i))
            if found is not None:
                return found
        return None

    def _read_word(self, x: np.ndarray, words: int) -> Optional[int]:
        """The 23 bits of the last ``words`` words of ``x``, added up bit
        by bit, or None when they don't read as DCS."""
        # Halfway between the two levels (two-means, from the median; on
        # the sorted samples each round is a search and two sums).
        ordered = np.sort(x)
        sums = np.concatenate(([0.0], np.cumsum(ordered)))
        n = ordered.size
        middle = float(ordered[n // 2])
        upper = lower = middle
        for _ in range(3):
            below = int(np.searchsorted(ordered, middle, side="right"))
            if below == 0 or below == n:
                return None
            lower = float(sums[below]) / below
            upper = float(sums[n] - sums[below]) / (n - below)
            middle = 0.5 * (upper + lower)
        if not upper - lower > 1e-9:
            return None
        y = x - middle
        # Bit timing: crossings of the middle fall on bit boundaries.
        sign = np.signbit(y)
        at = np.flatnonzero(sign[:-1] != sign[1:])
        if at.size < 2 * words:
            return None
        crossings = at + y[at] / (y[at] - y[at + 1])
        turns = np.exp(2j * np.pi * crossings / self._spb).sum()
        if abs(turns) < self.MIN_ALIGN * at.size:
            return None
        offset = (math.atan2(turns.imag, turns.real) / (2 * math.pi)) % 1.0
        offset *= self._spb
        complete = int((y.size - 1 - offset) // self._spb)
        need = words * _BITS
        if complete < need:
            return None
        centers = offset + (np.arange(complete - need, complete) + 0.5) * self._spb
        points = centers[:, None] + self._probe[None, :]
        soft = np.interp(points, np.arange(y.size), y).mean(axis=1)
        soft = soft.reshape(words, _BITS)
        bits = soft.sum(axis=0) > 0.0
        if np.count_nonzero((soft > 0.0) != bits[None, :]) > self.MAX_ERRORS * words:
            return None
        return int(np.dot(bits.astype(np.int64), 1 << np.arange(_BITS, dtype=np.int64)))

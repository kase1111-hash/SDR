"""
Text modes: Morse code (CW), RTTY, PSK31 and ASCII FSK, from text to the
signal that carries it.

A :class:`TextSignal` is built once from the text and its
:class:`TextSettings`; it is a timeline (marks, bits or symbols with their
times), so any part of it can be worked out exactly, at any sample rate,
without making the whole signal first:

* :meth:`TextSignal.audio`: what a receiver plays (Preview, Save as WAV,
  a radio's data input): the CW tone, the two FSK tones, the PSK31 tone;
* :meth:`TextSignal.baseband`: the emission as complex I/Q centred on the
  frequency it is sent on (Save as SigMF or cf32, and the transmitter);
* :class:`TextTxSource`: the emission streamed block by block at the
  transmit rate (``read(n)``, for :class:`~sdr_module.gui.transmitter.TxJob`),
  on the carrier itself or as audio on an FM carrier.

Modes
-----

* **CW** (Morse code): PARIS timing, a dit lasting 1.2/WPM s; one dit
  between the elements of a letter, three between letters and seven between
  words; raised-cosine key edges of 5 ms (no key clicks). Letters, digits
  and ``. , ? ' ! / ( ) & : ; = + - _ " $ @``; prosigns are written in angle
  brackets (``<SK>``, ``<AR>``, ``<KN>``): their letters go out as one
  character, without the letter gaps.
* **RTTY**: ITA2 (Baudot) at 45.45 Bd with a 170 Hz shift by default, the
  amateur standard: a start bit (space), five data bits (first bit first)
  and 1.5 stop bits (mark); mark is the higher frequency. LTRS and FIGS are
  put in by themselves (US-TTY figures by default: digits and
  ``- ? : $ ! & # ' ( ) " . , / ;``); after a space sent in figures the
  shift is repeated before the next figure, so receivers that unshift on
  space print it right too. Letters go out as capitals and a new line as
  CR LF. A short mark carrier and two LTRS go first, so a receiver is in
  step before the text.
* **PSK31** (G3PLX): binary PSK at 31.25 Bd with the full 128-character
  Varicode (case-sensitive: every ASCII character), two zero bits between
  characters; a zero bit is a phase reversal, shaped by a cosine through
  zero at the symbol boundary, so the signal stays about 60 Hz wide. About a
  second of reversals goes first (the receiver's sync), a short steady
  carrier last. A new line is CR LF.
* **ASCII** FSK: asynchronous 8N1 like a serial port (a start bit, eight
  data bits least significant first, a stop bit) of the ASCII text, sent
  as it is (a new line is LF). Default 300 Bd with 200 Hz shift on 1070 /
  1270 Hz (Bell 103, which ``minimodem --rx 300`` reads).

Nothing is dropped silently: :func:`check_text` (and every signal) lists
the characters a mode can't send, with how often each occurs. Curly
quotes, dashes and the ellipsis are sent as their ASCII forms.

Emissions
---------

``"direct"`` puts the keyed carrier (CW, A1A), the FSK (F1B) or the PSK
(G1B) on the transmit frequency itself, centred on it: a CW or SSB receiver
hears it (an SSB receiver in USB tuned 2.21 kHz below an RTTY signal hears
the usual 2125/2295 Hz tones). ``"fm"`` sends the audio on an FM carrier
(MCW, F2A, for CW; F2B for the data modes), which FM receivers hear.

NumPy only; no Qt.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import wave
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Modes and their settings
# --------------------------------------------------------------------------- #

MODE_CW = "CW"
MODE_RTTY = "RTTY"
MODE_PSK31 = "PSK31"
MODE_ASCII = "ASCII"
TEXT_MODES = (MODE_CW, MODE_RTTY, MODE_PSK31, MODE_ASCII)

#: How each mode is named in sentences.
MODE_NAMES = {
    MODE_CW: "CW",
    MODE_RTTY: "RTTY",
    MODE_PSK31: "PSK31",
    MODE_ASCII: "ASCII FSK",
}

EMISSION_DIRECT = "direct"
EMISSION_FM = "fm"
EMISSIONS = (EMISSION_DIRECT, EMISSION_FM)

RTTY_BAUD = 45.45
RTTY_SHIFT_HZ = 170.0
PSK31_BAUD = 31.25
ASCII_BAUD = 300.0
ASCII_SHIFT_HZ = 200.0

#: Choices offered for each mode (the encoder takes any value in range).
RTTY_BAUDS = (45.45, 50.0, 75.0, 100.0)
RTTY_SHIFTS = (170.0, 200.0, 425.0, 850.0)
ASCII_BAUDS = (110.0, 300.0, 600.0, 1200.0)
ASCII_SHIFTS = (170.0, 200.0, 425.0, 850.0, 1000.0)

WPM_RANGE = (5.0, 60.0)
BAUD_RANGE = (10.0, 1200.0)
SHIFT_RANGE = (20.0, 1000.0)
AUDIO_RANGE = (100.0, 3500.0)

#: Peak deviation of the FM emission (narrow FM, as the station ID).
FM_DEVIATION_HZ = 2500.0
#: Unmodulated FM carrier before and after the text, so an FM receiver's
#: squelch is open before the first element and stays open through the last.
FM_LEAD_S = 0.3
FM_TAIL_S = 0.15

# Key-click-free keying: raised-cosine edges this long.
_RISE_S = 0.005
# Key up before the first element and after the last (CW): a word space
# (seven dits), so a station ID sent just before or after the text is heard
# as a word of its own ("DE W1AW CQ", not "DE W1AWCQ").
_CW_GAP_DITS = 7
# Mark carrier before the first start bit and after the last stop bit.
_FSK_LEAD_S = 0.3
_RTTY_TAIL_S = 0.2
_ASCII_TAIL_S = 0.1
# RTTY: LTRS characters sent before the text.
_RTTY_PREAMBLE = 2
# PSK31: reversals before the text (about a second), steady carrier after.
_PSK_PREAMBLE_BITS = 32
_PSK_POSTAMBLE_BITS = 16

#: Most characters one text may hold (the Transmit Text box allows this).
MAX_TEXT_CHARS = 2000


@dataclass(frozen=True)
class TextSettings:
    """How a text is sent.

    ``wpm``: CW speed. ``baud`` and ``shift_hz``: RTTY and ASCII FSK (PSK31
    always runs at 31.25 Bd). ``audio_hz``: where the signal sits in the
    audio (the CW tone, the centre between the FSK tones, the PSK31 tone);
    it doesn't change what is transmitted on the carrier. ``figures``:
    RTTY's figures case, "us" (US TTY, what amateur software uses) or
    "ita2" (international).
    """

    mode: str = MODE_RTTY
    wpm: float = 20.0
    baud: float = RTTY_BAUD
    shift_hz: float = RTTY_SHIFT_HZ
    audio_hz: float = 2210.0
    figures: str = "us"

    @classmethod
    def for_mode(cls, mode: str, **values) -> "TextSettings":
        """The usual settings of ``mode`` (CW 20 WPM on a 700 Hz tone; RTTY
        45.45 Bd, 170 Hz on 2125/2295 Hz; PSK31 on 1000 Hz; ASCII 300 Bd,
        200 Hz on 1070/1270 Hz), with ``values`` changed."""
        mode = normalize_mode(mode)
        base = {
            MODE_CW: cls(MODE_CW, audio_hz=700.0),
            MODE_RTTY: cls(MODE_RTTY),
            MODE_PSK31: cls(MODE_PSK31, baud=PSK31_BAUD, shift_hz=0.0, audio_hz=1000.0),
            MODE_ASCII: cls(
                MODE_ASCII, baud=ASCII_BAUD, shift_hz=ASCII_SHIFT_HZ, audio_hz=1170.0
            ),
        }[mode]
        return replace(base, **values).clamped()

    def clamped(self) -> "TextSettings":
        """These settings with every value in its range (a value that is not
        a number takes the bottom of its range). Both FSK tones stay in
        :data:`AUDIO_RANGE` (see :func:`audio_bounds`)."""

        def clamp(value: float, bounds: Tuple[float, float]) -> float:
            value = float(value)
            if math.isnan(value):
                return float(bounds[0])
            return float(min(max(value, bounds[0]), bounds[1]))

        mode = normalize_mode(self.mode)
        baud = PSK31_BAUD if mode == MODE_PSK31 else clamp(self.baud, BAUD_RANGE)
        shift = 0.0 if mode == MODE_PSK31 else clamp(self.shift_hz, SHIFT_RANGE)
        return TextSettings(
            mode,
            clamp(self.wpm, WPM_RANGE),
            baud,
            shift,
            clamp(self.audio_hz, audio_bounds(mode, shift)),
            "ita2" if str(self.figures).lower() == "ita2" else "us",
        )

    def audio_bounds(self) -> Tuple[float, float]:
        """The range ``audio_hz`` may take with these settings."""
        return audio_bounds(self.mode, self.shift_hz)

    @property
    def symbol_rate(self) -> float:
        """Symbols per second: the baud rate, or for CW the dit rate."""
        if self.mode == MODE_CW:
            return self.wpm / 1.2
        return PSK31_BAUD if self.mode == MODE_PSK31 else self.baud

    def describe(self) -> str:
        """``"RTTY 45.45 Bd, 170 Hz shift"``, ``"CW 20 WPM"``, ``"PSK31"``,
        ``"ASCII FSK 300 Bd, 200 Hz shift"``."""
        name = MODE_NAMES[self.mode]
        if self.mode == MODE_CW:
            return f"{name} {format_number(self.wpm)} WPM"
        if self.mode == MODE_PSK31:
            return name
        return f"{name} {format_number(self.baud)} Bd, {format_number(self.shift_hz)} Hz shift"

    def tones(self) -> Tuple[float, ...]:
        """The audio tones: (tone,) for CW and PSK31, (space, mark) for FSK."""
        if self.mode in (MODE_RTTY, MODE_ASCII):
            half = self.shift_hz / 2.0
            return (self.audio_hz - half, self.audio_hz + half)
        return (self.audio_hz,)


def normalize_mode(mode: str) -> str:
    """``"rtty"`` -> ``"RTTY"``; also "MORSE", "PSK", "ASCII FSK"."""
    text = str(mode or "").strip().upper()
    aliases = {
        "MORSE": MODE_CW,
        "CW (MORSE)": MODE_CW,
        "PSK": MODE_PSK31,
        "BPSK31": MODE_PSK31,
        "ASCII FSK": MODE_ASCII,
        "FSK": MODE_ASCII,
    }
    text = aliases.get(text, text)
    if text not in TEXT_MODES:
        raise ValueError(f"unknown text mode {mode!r}")
    return text


def format_number(value: float) -> str:
    """``45.45 -> "45.45"``, ``300.0 -> "300"``."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def audio_bounds(mode: str, shift_hz: float = 0.0) -> Tuple[float, float]:
    """The range of the audio tone (CW, PSK31), or of the centre between
    the two FSK tones (RTTY, ASCII), that keeps every tone in
    :data:`AUDIO_RANGE`: the centre stays at least half the shift from
    either end (a 100 Hz centre with a 1000 Hz shift would put space at
    -400 Hz, which plays as 400 Hz)."""
    low, high = AUDIO_RANGE
    if normalize_mode(mode) in (MODE_RTTY, MODE_ASCII):
        half = min(max(0.0, float(shift_hz)), SHIFT_RANGE[1]) / 2.0
        # (Whole hertz, as the Transmit Text window sets them.)
        return (float(math.ceil(low + half)), float(math.floor(high - half)))
    return (low, high)


# --------------------------------------------------------------------------- #
# Character sets
# --------------------------------------------------------------------------- #

#: Typographic characters sent as their ASCII forms (pasted text is full of
#: them); everything else a mode lacks is reported.
_TYPOGRAPHIC = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "…": "...",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
}

# ITA2, indexed by the code with its first-sent bit as bit 0. None: no
# character (blank, the shifts, or unassigned).
_LTRS = 0b11111
_FIGS = 0b11011
_CR = 0b01000
_LF = 0b00010
_SPACE = 0b00100

ITA2_LETTERS: Tuple[Optional[str], ...] = (
    None, "E", "\n", "A", " ", "S", "I", "U",
    "\r", "D", "R", "J", "N", "F", "C", "K",
    "T", "Z", "L", "W", "H", "Y", "P", "Q",
    "O", "B", "G", None, "M", "X", "V", None,
)  # fmt: skip

#: Figures case of US teleprinters (what amateur RTTY software uses).
ITA2_FIGURES_US: Tuple[Optional[str], ...] = (
    None, "3", "\n", "-", " ", "\a", "8", "7",
    "\r", "$", "4", "'", ",", "!", ":", "(",
    "5", '"', ")", "2", "#", "6", "0", "1",
    "9", "?", "&", None, ".", "/", ";", None,
)  # fmt: skip

#: International ITA2 figures (D is "who are you", F, G and H unassigned).
ITA2_FIGURES_INTL: Tuple[Optional[str], ...] = (
    None, "3", "\n", "-", " ", "'", "8", "7",
    "\r", None, "4", "\a", ",", None, ":", "(",
    "5", "+", ")", "2", None, "6", "0", "1",
    "9", "?", None, None, ".", "/", "=", None,
)  # fmt: skip

#: PSK31 Varicode (G3PLX), indexed by ASCII code 0-127.
_VARICODE_CODES: Tuple[str, ...] = (
    "1010101011", "1011011011", "1011101101", "1101110111",  # NUL SOH STX ETX
    "1011101011", "1101011111", "1011101111", "1011111101",  # EOT ENQ ACK BEL
    "1011111111", "11101111", "11101", "1101101111",  # BS HT LF VT
    "1011011101", "11111", "1101110101", "1110101011",  # FF CR SO SI
    "1011110111", "1011110101", "1110101101", "1110101111",  # DLE DC1 DC2 DC3
    "1101011011", "1101101011", "1101101101", "1101010111",  # DC4 NAK SYN ETB
    "1101111011", "1101111101", "1110110111", "1101010101",  # CAN EM SUB ESC
    "1101011101", "1110111011", "1011111011", "1101111111",  # FS GS RS US
    "1", "111111111", "101011111", "111110101",  # space ! " #
    "111011011", "1011010101", "1010111011", "101111111",  # $ % & '
    "11111011", "11110111", "101101111", "111011111",  # ( ) * +
    "1110101", "110101", "1010111", "110101111",  # , - . /
    "10110111", "10111101", "11101101", "11111111",  # 0 1 2 3
    "101110111", "101011011", "101101011", "110101101",  # 4 5 6 7
    "110101011", "110110111", "11110101", "110111101",  # 8 9 : ;
    "111101101", "1010101", "111010111", "1010101111",  # < = > ?
    "1010111101", "1111101", "11101011", "10101101",  # @ A B C
    "10110101", "1110111", "11011011", "11111101",  # D E F G
    "101010101", "1111111", "111111101", "101111101",  # H I J K
    "11010111", "10111011", "11011101", "10101011",  # L M N O
    "11010101", "111011101", "10101111", "1101111",  # P Q R S
    "1101101", "101010111", "110110101", "101011101",  # T U V W
    "101110101", "101111011", "1010101101", "111110111",  # X Y Z [
    "111101111", "111111011", "1010111111", "101101101",  # \ ] ^ _
    "1011011111", "1011", "1011111", "101111",  # ` a b c
    "101101", "11", "111101", "1011011",  # d e f g
    "101011", "1101", "111101011", "10111111",  # h i j k
    "11011", "111011", "1111", "111",  # l m n o
    "111111", "110111111", "10101", "10111",  # p q r s
    "101", "110111", "1111011", "1101011",  # t u v w
    "11011111", "1011101", "111010101", "1010110111",  # x y z {
    "110111011", "1010110101", "1011010111", "1110110101",  # | } ~ DEL
)  # fmt: skip

#: Character -> Varicode (all 128 ASCII characters).
VARICODE: Dict[str, str] = {chr(i): code for i, code in enumerate(_VARICODE_CODES)}


def _morse_table() -> Dict[str, str]:
    """Character -> Morse code, from the decoder's own table (dsp/morse.py),
    so what is sent is what the app's Morse decoder reads."""
    from ..dsp.morse import MORSE_CODE

    return {char: code for code, char in MORSE_CODE.items() if len(char) == 1}


MORSE_CODES: Dict[str, str] = _morse_table()


@dataclass
class TextCheck:
    """What a mode makes of a text.

    ``sent``: the text as a receiver prints it (capitals in CW and RTTY,
    ASCII forms of curly quotes, the characters that can't be sent left
    out). ``characters``: how many characters go out (a prosign counts
    once). ``unsupported``: each character that can't be sent and how often
    it occurs, in the order they first appear.
    """

    mode: str
    sent: str = ""
    characters: int = 0
    unsupported: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unsupported

    @property
    def skipped(self) -> int:
        """How many characters are left out (counting repeats)."""
        return sum(self.unsupported.values())

    def describe_unsupported(self, limit: int = 8) -> str:
        """``"@ (2x), %, é"``: the characters left out (repeats counted);
        control characters by name ("tab")."""
        parts = []
        for char, count in list(self.unsupported.items())[:limit]:
            name = char_name(char)
            parts.append(f"{name} ({count}x)" if count > 1 else name)
        if len(self.unsupported) > limit:
            parts.append("...")
        return ", ".join(parts)


def char_name(char: str) -> str:
    """How to show ``char`` in a message: itself, or "tab", "new line"..."""
    names = {"\t": "tab", "\n": "new line", "\r": "return", " ": "space"}
    if char in names:
        return names[char]
    if ord(char) < 32 or ord(char) == 127:
        return f"control character {ord(char):#04x}"
    return char


def _normalize(text: str) -> str:
    """Typographic characters as ASCII; a new line is "\\n" whatever it was
    typed as."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(_TYPOGRAPHIC.get(c, c) for c in text)


def _note(unsupported: Dict[str, int], char: str) -> None:
    unsupported[char] = unsupported.get(char, 0) + 1


# --------------------------------------------------------------------------- #
# Text -> symbols
# --------------------------------------------------------------------------- #


def ita2_codes(text: str, figures: str = "us") -> Tuple[List[int], TextCheck]:
    """ITA2 codes of ``text`` with the LTRS and FIGS shifts it needs (the
    first character is preceded by its shift), and what was sent.

    A new line is CR LF; a tab is sent as a space; letters as capitals.
    After a space (or CR, LF) sent in figures, the next figure gets FIGS
    again (receivers that "unshift on space" are then in letters).
    """
    table = ITA2_FIGURES_INTL if str(figures).lower() == "ita2" else ITA2_FIGURES_US
    letters = {c: i for i, c in enumerate(ITA2_LETTERS) if c is not None}
    figs = {c: i for i, c in enumerate(table) if c is not None}
    neutral = {" ": _SPACE, "\r": _CR}  # in both cases
    check = TextCheck(MODE_RTTY)
    codes: List[int] = []
    sent: List[str] = []
    shift: Optional[int] = None  # unknown at the start
    unsure = False  # a space was sent in figures
    for char in _normalize(text):
        if char == "\t":
            char = " "
        if char == "\n":
            codes += [_CR, _LF]
            sent.append("\n")
            check.characters += 1
            if shift == _FIGS:
                unsure = True
            continue
        if char in neutral:
            codes.append(neutral[char])
            sent.append(char)
            check.characters += 1
            if shift == _FIGS:
                unsure = True
            continue
        # (Only ASCII letters: "\u00df".upper() is "SS", "\u0131".upper() "I".)
        upper = char.upper() if char.isascii() else char
        if upper in letters and upper.isalpha():
            if shift != _LTRS:
                codes.append(_LTRS)
                shift = _LTRS
            unsure = False
            codes.append(letters[upper])
            sent.append(upper)
            check.characters += 1
        elif char in figs and char not in ("\n", " ", "\r"):
            if shift != _FIGS or unsure:
                codes.append(_FIGS)
                shift = _FIGS
            unsure = False
            codes.append(figs[char])
            sent.append(char)
            check.characters += 1
        else:
            _note(check.unsupported, char)
    check.sent = "".join(sent)
    return codes, check


def varicode_bits(text: str) -> Tuple[List[int], TextCheck]:
    """PSK31 bits of ``text``: each character's Varicode followed by two
    zeros; a new line is CR LF. And what was sent."""
    check = TextCheck(MODE_PSK31)
    bits: List[int] = []
    sent: List[str] = []
    for char in _normalize(text):
        chars = "\r\n" if char == "\n" else char
        if all(c in VARICODE for c in chars):
            for c in chars:
                bits += [int(b) for b in VARICODE[c]] + [0, 0]
            sent.append(char)
            check.characters += 1
        else:
            _note(check.unsupported, char)
    check.sent = "".join(sent)
    return bits, check


def _varicode_word_ends(text: str) -> List[int]:
    """The bits of :func:`varicode_bits` sent by the end of each space, tab
    or new line (its two zeros included): where a word ends."""
    ends: List[int] = []
    count = 0
    for char in _normalize(text):
        chars = "\r\n" if char == "\n" else char
        if all(c in VARICODE for c in chars):
            count += sum(len(VARICODE[c]) + 2 for c in chars)
            if char in (" ", "\t", "\n"):
                ends.append(count)
    return ends


def ascii_bytes(text: str) -> Tuple[bytes, TextCheck]:
    """The ASCII bytes of ``text`` (sent as they are; a new line is LF) and
    what was sent."""
    check = TextCheck(MODE_ASCII)
    out = bytearray()
    sent: List[str] = []
    for char in _normalize(text):
        if ord(char) < 128:
            out.append(ord(char))
            sent.append(char)
            check.characters += 1
        else:
            _note(check.unsupported, char)
    check.sent = "".join(sent)
    return bytes(out), check


def morse_elements(text: str) -> Tuple[List[List[str]], TextCheck]:
    """``text`` as words of Morse characters (each a string of "." and
    "-"), and what was sent. Prosigns in angle brackets (``<SK>``) are one
    character made of their letters' elements; any white space is a word
    gap; letters are sent as capitals."""
    check = TextCheck(MODE_CW)
    words: List[List[str]] = []
    sent_words: List[str] = []
    # Capitals (ASCII letters only: other letters are reported, not changed).
    normalized = "".join(c.upper() if c.isascii() else c for c in _normalize(text))
    for raw_word in normalized.split():
        word: List[str] = []
        shown: List[str] = []
        i = 0
        while i < len(raw_word):
            char = raw_word[i]
            if char == "<":
                end = raw_word.find(">", i + 1)
                inner = raw_word[i + 1 : end] if end > i + 1 else ""
                if inner and all(c in MORSE_CODES for c in inner):
                    word.append("".join(MORSE_CODES[c] for c in inner))
                    shown.append(f"<{inner}>")
                    check.characters += 1
                    i = end + 1
                    continue
            if char in MORSE_CODES:
                word.append(MORSE_CODES[char])
                shown.append(char)
                check.characters += 1
            else:
                _note(check.unsupported, char)
            i += 1
        if word:
            words.append(word)
            sent_words.append("".join(shown))
    check.sent = " ".join(sent_words)
    return words, check


def check_text(text: str, mode: str, figures: str = "us") -> TextCheck:
    """What ``mode`` makes of ``text``: what is sent, how many characters,
    and which characters it can't send (see :class:`TextCheck`)."""
    mode = normalize_mode(mode)
    if mode == MODE_CW:
        return morse_elements(text)[1]
    if mode == MODE_RTTY:
        return ita2_codes(text, figures)[1]
    if mode == MODE_PSK31:
        return varicode_bits(text)[1]
    return ascii_bytes(text)[1]


def remove_unsupported(text: str, mode: str, figures: str = "us") -> str:
    """``text`` without the characters ``mode`` can't send, everything else
    kept as typed (curly quotes, new lines, small letters). In CW the
    brackets of a good prosign (``<SK>``) stay; those of a broken one go."""
    mode = normalize_mode(mode)
    out: List[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if mode == MODE_CW and char == "<":
            end = text.find(">", i + 1)
            inner = text[i + 1 : end] if end > i + 1 else ""
            if (
                inner
                and inner.isascii()
                and all(c in MORSE_CODES for c in inner.upper())
            ):
                out.append(text[i : end + 1])
                i = end + 1
                continue
        if check_text(char, mode, figures).ok:
            out.append(char)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Symbols -> signal
# --------------------------------------------------------------------------- #


def _rise(x: np.ndarray, rise: np.ndarray) -> np.ndarray:
    """Raised-cosine edge: 0 at ``x`` = 0, 1 from ``x`` = ``rise`` on."""
    ratio = np.clip(x / np.maximum(rise, 1e-12), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * ratio)


class _Keyed:
    """CW: key-down intervals with raised-cosine edges (inside each element,
    so the timing is exact)."""

    def __init__(self, starts: Sequence[float], ends: Sequence[float]):
        self.starts = np.asarray(starts, np.float64)
        self.ends = np.asarray(ends, np.float64)
        self.rises = np.minimum(_RISE_S, (self.ends - self.starts) / 2.0)

    def envelope(self, t: np.ndarray) -> np.ndarray:
        out = np.zeros(len(t))
        if not len(self.starts):
            return out
        k = np.searchsorted(self.starts, t, side="right") - 1
        kc = np.clip(k, 0, len(self.starts) - 1)
        inside = (k >= 0) & (t < self.ends[kc])
        rise = self.rises[kc]
        env = np.minimum(
            _rise(t - self.starts[kc], rise), _rise(self.ends[kc] - t, rise)
        )
        out[inside] = env[inside]
        return out


class _Fsk:
    """FSK: segments at mark (+shift/2) or space (-shift/2), phase
    continuous; the carrier rises and falls with raised-cosine edges."""

    def __init__(self, segments: Sequence[Tuple[float, float]], duration: float):
        # segments: (frequency offset Hz, seconds)
        freqs = np.array([f for f, _d in segments], np.float64)
        lengths = np.array([d for _f, d in segments], np.float64)
        self.starts = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
        self.freqs = freqs
        phase_steps = 2.0 * np.pi * freqs * lengths
        self.phases = np.concatenate(([0.0], np.cumsum(phase_steps)[:-1]))
        self.duration = float(duration)

    def phase(self, t: np.ndarray) -> np.ndarray:
        k = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, None)
        return self.phases[k] + 2.0 * np.pi * self.freqs[k] * (t - self.starts[k])

    def envelope(self, t: np.ndarray) -> np.ndarray:
        rise = np.full(len(t), _RISE_S)
        env = np.minimum(_rise(t, rise), _rise(self.duration - t, rise))
        env[(t < 0.0) | (t >= self.duration)] = 0.0
        return env


class _Psk:
    """PSK31 envelope (real: +1/-1 with cosine-shaped reversals through
    zero at the symbol boundaries), one symbol of rise and of fall."""

    # Segment kinds.
    STEADY, REVERSE, RISE, FALL = 0, 1, 2, 3

    def __init__(self, bits: Sequence[int], baud: float):
        self.symbol = 1.0 / float(baud)
        signs: List[float] = [1.0]  # the sign before each segment
        kinds: List[int] = [self.RISE]
        sign = 1.0
        for bit in bits:
            signs.append(sign)
            if bit:
                kinds.append(self.STEADY)
            else:
                kinds.append(self.REVERSE)
                sign = -sign
        signs.append(sign)
        kinds.append(self.FALL)
        self.signs = np.asarray(signs, np.float64)
        self.kinds = np.asarray(kinds, np.int8)
        self.duration = len(kinds) * self.symbol

    def envelope(self, t: np.ndarray) -> np.ndarray:
        k = np.floor(t / self.symbol).astype(np.int64)
        valid = (k >= 0) & (k < len(self.kinds))
        kc = np.clip(k, 0, len(self.kinds) - 1)
        u = t / self.symbol - kc
        sign = self.signs[kc]
        kind = self.kinds[kc]
        cos = np.cos(np.pi * u)
        out = np.where(
            kind == self.REVERSE,
            sign * cos,
            np.where(
                kind == self.RISE,
                sign * (0.5 - 0.5 * cos),
                np.where(kind == self.FALL, sign * (0.5 + 0.5 * cos), sign),
            ),
        )
        out[~valid] = 0.0
        return out


def _uart_segments(
    frames: Sequence[int],
    data_bits: int,
    stop_bits: float,
    baud: float,
    shift: float,
    lead_s: float,
    tail_s: float,
) -> Tuple[List[Tuple[float, float]], float]:
    """FSK segments (frequency, seconds) of asynchronous frames: idle mark,
    then per frame a start bit (space), the data bits (first bit first) and
    the stop bits (mark); idle mark after. Runs of one tone are merged."""
    bit = 1.0 / float(baud)
    mark, space = shift / 2.0, -shift / 2.0
    raw: List[Tuple[float, float]] = [(mark, lead_s)]
    for value in frames:
        raw.append((space, bit))
        for k in range(data_bits):
            raw.append((mark if (value >> k) & 1 else space, bit))
        raw.append((mark, stop_bits * bit))
    raw.append((mark, tail_s))
    merged: List[Tuple[float, float]] = []
    for freq, length in raw:
        if length <= 0:
            continue
        if merged and merged[-1][0] == freq:
            merged[-1] = (freq, merged[-1][1] + length)
        else:
            merged.append((freq, length))
    return merged, float(sum(length for _f, length in merged))


class TextSignal:
    """``text`` sent in a text mode (see the module docstring).

    Attributes:
        text: The text as given.
        settings: The (clamped) :class:`TextSettings`.
        check: What is sent and what can't be (:class:`TextCheck`).
        duration_s: Length of the whole signal (lead-in and tail included).
    """

    def __init__(self, text: str, settings: TextSettings):
        self.text = str(text)
        self.settings = settings.clamped()
        mode = self.settings.mode
        self._keyed: Optional[_Keyed] = None
        self._fsk: Optional[_Fsk] = None
        self._psk: Optional[_Psk] = None
        # Where the text may stop for a station ID (seconds): the ends of
        # its words (see :attr:`breaks`).
        breaks: List[float] = []
        if mode == MODE_CW:
            words, self.check = morse_elements(self.text)
            self._keyed, self.duration_s = self._build_cw(words, breaks)
        elif mode == MODE_PSK31:
            bits, self.check = varicode_bits(self.text)
            bits = [0] * _PSK_PREAMBLE_BITS + bits + [1] * _PSK_POSTAMBLE_BITS
            self._psk = _Psk(bits, PSK31_BAUD)
            self.duration_s = self._psk.duration
            # After the two zeros that end a space or a new line (the RISE
            # symbol and the preamble go first).
            breaks += [
                (1 + _PSK_PREAMBLE_BITS + count) * self._psk.symbol
                for count in _varicode_word_ends(self.text)
            ]
        else:
            if mode == MODE_RTTY:
                codes, self.check = ita2_codes(self.text, self.settings.figures)
                frames = [_LTRS] * _RTTY_PREAMBLE + codes
                if codes and codes[0] == _LTRS:
                    frames = frames[1:]  # (two LTRS in all)
                frame_bits, word_ends = 7.5, (_SPACE, _LF)
                segments, self.duration_s = _uart_segments(
                    frames,
                    5,
                    1.5,
                    self.settings.baud,
                    self.settings.shift_hz,
                    _FSK_LEAD_S,
                    _RTTY_TAIL_S,
                )
            else:
                data, self.check = ascii_bytes(self.text)
                frames = list(data)
                frame_bits, word_ends = 10.0, (0x20, 0x0A)
                segments, self.duration_s = _uart_segments(
                    frames,
                    8,
                    1.0,
                    self.settings.baud,
                    self.settings.shift_hz,
                    _FSK_LEAD_S,
                    _ASCII_TAIL_S,
                )
            self._fsk = _Fsk(segments, self.duration_s)
            # At the end of the stop bit of a space or a line feed.
            frame_s = frame_bits / self.settings.baud
            breaks += [
                _FSK_LEAD_S + (i + 1) * frame_s
                for i, code in enumerate(frames)
                if code in word_ends
            ]
        self._breaks = np.asarray(breaks, np.float64)

    def _build_cw(
        self, words: List[List[str]], breaks: List[float]
    ) -> Tuple[_Keyed, float]:
        dit = 1.2 / self.settings.wpm
        lead = _CW_GAP_DITS * dit
        starts: List[float] = []
        ends: List[float] = []
        units = 0
        for w, word in enumerate(words):
            if w:
                # The middle of the word space: key up on both sides.
                breaks.append(lead + (units + 3.5) * dit)
                units += 7
            for c, code in enumerate(word):
                if c:
                    units += 3
                for e, element in enumerate(code):
                    if e:
                        units += 1
                    length = 1 if element == "." else 3
                    starts.append(lead + units * dit)
                    units += length
                    ends.append(lead + units * dit)
        return _Keyed(starts, ends), lead + (units + _CW_GAP_DITS) * dit

    # ------------------------------------------------------------- facts

    @property
    def mode(self) -> str:
        return self.settings.mode

    @property
    def characters(self) -> int:
        """Characters sent."""
        return self.check.characters

    @property
    def empty(self) -> bool:
        """Nothing to send (no character the mode can send)."""
        return self.check.characters == 0

    def tx_mode(self, emission: str = EMISSION_DIRECT) -> str:
        """The mode the legality check knows the emission by: "CW" or
        "DATA" on the carrier, "FM" on an FM carrier."""
        if emission == EMISSION_FM:
            return "FM"
        return "CW" if self.mode == MODE_CW else "DATA"

    def occupied_bandwidth_hz(
        self, emission: str = EMISSION_DIRECT, deviation_hz: float = FM_DEVIATION_HZ
    ) -> float:
        """What the emission occupies (for the legality check, which tests
        both its edges), never less than its 99% power bandwidth whatever
        the text: CW at least five times the keying rate (and 150 Hz), wider
        where runs of dits need it (:func:`cw_bandwidth_hz`); FSK from the
        99% bandwidth of alternating keying, the widest a text makes
        (:func:`fsk_bandwidth_hz`: 330 Hz for standard RTTY); PSK31 60 Hz
        (it measures about 45 Hz); on FM, Carson's rule over the highest
        audio frequency."""
        s = self.settings
        if emission == EMISSION_FM:
            top = max(s.tones())
            if s.mode in (MODE_RTTY, MODE_ASCII):
                top += s.baud / 2.0
            elif s.mode == MODE_PSK31:
                top += PSK31_BAUD
            return 2.0 * (abs(float(deviation_hz)) + top)
        if s.mode == MODE_CW:
            return cw_bandwidth_hz(s.wpm)
        if s.mode == MODE_PSK31:
            return 60.0
        return fsk_bandwidth_hz(s.mode, s.baud, s.shift_hz)

    def describe(self) -> str:
        return self.settings.describe()

    @property
    def breaks(self) -> np.ndarray:
        """Where the text can stop for a station ID and go on after it
        (seconds from the start, in order): the ends of its words. CW: the
        middle of each word space (key up on both sides); RTTY and ASCII:
        the end of the stop bit of a space or a line feed; PSK31: after the
        two zeros that end a space or a new line."""
        return self._breaks.copy()

    # ------------------------------------------------- stopping for an ID

    @property
    def pause_tail_s(self) -> float:
        """How the text ends at a break before a station ID (seconds):
        PSK31 idles on for one more phase reversal (a zero bit: nothing is
        printed), so a receiver reads the last one of the text, then half a
        reversal, whose cosine takes the carrier smoothly to zero; the
        others stop at the break."""
        return 1.5 / PSK31_BAUD if self.mode == MODE_PSK31 else 0.0

    @property
    def resume_lead_in_s(self) -> float:
        """The lead-in sent after a station ID, before the text goes on
        (seconds), so that a receiver falls back in step: what the text
        itself begins with (RTTY: idle mark and two LTRS; ASCII: idle mark;
        PSK31: the phase reversals); none for CW (the word space does)."""
        if self.mode == MODE_CW:
            return 0.0
        if self.mode == MODE_PSK31:
            return (1 + _PSK_PREAMBLE_BITS) / PSK31_BAUD
        if self.mode == MODE_RTTY:
            return _FSK_LEAD_S + _RTTY_PREAMBLE * 7.5 / self.settings.baud
        return _FSK_LEAD_S

    def lead_in(self) -> "TextSignal":
        """The lead-in on its own (see :attr:`resume_lead_in_s`): the
        signal of no text, read up to :attr:`resume_lead_in_s`."""
        return TextSignal("", self.settings)

    # ------------------------------------------------------------ synthesis

    def envelope_phase(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Amplitude (real, -1..1) and phase (rad) of the emission at times
        ``t`` (seconds from the start); 0 outside the signal."""
        t = np.asarray(t, np.float64)
        if self._keyed is not None:
            return self._keyed.envelope(t), np.zeros(len(t))
        if self._psk is not None:
            return self._psk.envelope(t), np.zeros(len(t))
        assert self._fsk is not None
        return self._fsk.envelope(t), self._fsk.phase(t)

    def baseband(self, t: np.ndarray) -> np.ndarray:
        """The emission as complex baseband (complex128), centred on 0 Hz,
        at times ``t``."""
        env, phase = self.envelope_phase(t)
        if self._fsk is None:
            return env.astype(np.complex128)
        return env * np.exp(1j * phase)

    def audio(self, t: np.ndarray, audio_hz: Optional[float] = None) -> np.ndarray:
        """What a receiver plays (-1..1) at times ``t``: the signal moved up
        to ``audio_hz`` (default: the settings' tone)."""
        f = self.settings.audio_hz if audio_hz is None else float(audio_hz)
        t = np.asarray(t, np.float64)
        env, phase = self.envelope_phase(t)
        return env * np.sin(2.0 * np.pi * f * t + phase)

    def analytic(self, t: np.ndarray, carrier_hz: float) -> np.ndarray:
        """The emission moved to ``carrier_hz`` (complex128): the I/Q the
        encoder classes and the command line write."""
        t = np.asarray(t, np.float64)
        return self.baseband(t) * np.exp(2j * np.pi * float(carrier_hz) * t)

    def sample_count(self, rate: float) -> int:
        return int(round(self.duration_s * float(rate)))

    def times(self, rate: float, start: int = 0, count: Optional[int] = None):
        """Sample times (s) ``start``.. at ``rate`` (to the end by default)."""
        total = self.sample_count(rate)
        stop = total if count is None else min(total, start + int(count))
        return np.arange(start, max(start, stop), dtype=np.float64) / float(rate)

    def source(
        self,
        rate: float,
        emission: str = EMISSION_DIRECT,
        deviation_hz: float = FM_DEVIATION_HZ,
        amplitude: float = 0.9,
    ) -> "TextTxSource":
        """The emission streamed at ``rate`` (see :class:`TextTxSource`)."""
        return TextTxSource(self, rate, emission, deviation_hz, amplitude)


def build_text_signal(text: str, settings: TextSettings) -> TextSignal:
    return TextSignal(text, settings)


def _power_bandwidth(iq: np.ndarray, rate: float, fraction: float = 0.99) -> float:
    """The width (Hz) holding ``fraction`` of the power of ``iq``, with
    half the rest below it and half above (the ITU's occupied bandwidth)."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1.0 / float(rate)))
    cumulative = np.cumsum(spectrum) / max(float(np.sum(spectrum)), 1e-300)
    tail = (1.0 - fraction) / 2.0
    low = freqs[int(np.searchsorted(cumulative, tail))]
    high = freqs[min(len(freqs) - 1, int(np.searchsorted(cumulative, 1.0 - tail)))]
    return float(high - low)


@functools.lru_cache(maxsize=64)
def fsk_bandwidth_hz(mode: str, baud: float, shift_hz: float) -> float:
    """Occupied bandwidth (Hz) of RTTY or ASCII FSK at ``baud`` and
    ``shift_hz``, whatever the text: 10% more than the larger of the ITU's
    necessary bandwidth (baud + 1.2 x shift) and the 99% power bandwidth of
    alternating keying ("RY" in RTTY, "U" in ASCII: a change of tone every
    bit, which spreads the most power into the keying sidebands). Keyed FSK
    is wider than the ITU formula for some patterns (up to a third), and
    other texts measure up to about 1.5% wider than the larger of the two,
    hence the margin. Measured once per setting (a few ms) and kept."""
    settings = TextSettings.for_mode(mode, baud=baud, shift_hz=shift_hz)
    pattern = "RY" * 24 if settings.mode == MODE_RTTY else "U" * 48
    signal = TextSignal(pattern, settings)
    rate = max(8000.0, 16.0 * (settings.shift_hz + 2.0 * settings.baud))
    # The keying only (the idle mark before and after is a pure tone).
    start = _FSK_LEAD_S + _RTTY_PREAMBLE * 7.5 / settings.baud
    stop = signal.duration_s - _RTTY_TAIL_S
    t = np.arange(int(start * rate), int(stop * rate)) / rate
    measured = _power_bandwidth(signal.baseband(t), rate)
    itu = settings.baud + 1.2 * settings.shift_hz
    return float(1.1 * max(itu, measured))


@functools.lru_cache(maxsize=64)
def cw_bandwidth_hz(wpm: float) -> float:
    """Occupied bandwidth (Hz) of Morse code at ``wpm``, whatever the text:
    the ITU's necessary bandwidth on fading paths (five times the keying
    rate, at least 150 Hz for the 5 ms key edges; what the station ID
    declares), or 10% more than the 99% power bandwidth of the densest
    keying (runs of dits: "5", "H", "I", "E") where that is wider: from
    about 25 to 45 WPM, where the key edges take a larger part of each dit
    (200 Hz measured at 40 WPM, against 167 Hz). Measured once per speed
    (at most about 10 ms) and kept."""
    settings = TextSettings.for_mode(MODE_CW, wpm=wpm)
    itu = max(150.0, 5.0 * settings.symbol_rate)
    rate = 2000.0  # (the keying sidebands are far below 1% beyond 1 kHz)
    # Fewer characters at slow speeds: as many dits, measured as well.
    scale = min(1.0, settings.wpm / 20.0)
    measured = 0.0
    for char, count in (("5", 6), ("H", 8), ("I", 12), ("E", 20)):
        signal = TextSignal(char * max(2, int(round(count * scale))), settings)
        iq = signal.baseband(signal.times(rate))
        measured = max(measured, _power_bandwidth(iq, rate))
    return float(max(itu, 1.1 * measured))


# --------------------------------------------------------------------------- #
# Streaming (the transmitter, file export)
# --------------------------------------------------------------------------- #


def _ramp_up(k: np.ndarray, length: int) -> np.ndarray:
    """Raised-cosine rise over ``length`` samples, at sample(s) ``k``
    (1 from ``length`` on)."""
    x = np.clip((np.asarray(k, np.float64) + 0.5) / max(1, int(length)), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


class _Piece:
    """A stretch of what a :class:`TextTxSource` sends: the text from one of
    its samples on, or, around a stop for a station ID, the text's ending,
    the stop itself, the unmodulated FM carrier and the lead-in after it."""

    TEXT, TAIL, STOP, LEAD, LEAD_IN = range(5)

    __slots__ = ("kind", "length", "done", "start", "t", "value", "align")

    def __init__(
        self,
        kind: int,
        length: int,
        start: int = 0,
        t: float = 0.0,
        value: complex = 0j,
        align: complex = 1 + 0j,
    ):
        self.kind = kind
        self.length = max(0, int(length))
        self.done = 0
        self.start = int(start)  # TEXT: its first sample of the text
        self.t = float(t)  # the text's time at the stop (s)
        self.value = value  # the text's baseband there
        self.align = align  # phase (and sign) joining the lead-in to it


class TextTxSource:
    """A :class:`TextSignal` streamed as complex64 I/Q at ``rate``, a block
    at a time (``read(n)``; empty at the end), the way the transmit engine
    reads a job. Memory stays at one block whatever the length or rate.

    ``emission`` "direct": the emission on the carrier (0 Hz is the transmit
    frequency). "fm": the signal's audio on an FM carrier, ``deviation_hz``
    peak, with :data:`FM_LEAD_S` / :data:`FM_TAIL_S` of unmodulated carrier
    around it; the carrier rises and falls over 5 ms (raised cosine), so it
    never switches on or off in one sample. ``amplitude``: the digital
    level (headroom below full scale).

    A station ID can go into a long text without cutting a word: the
    transmit engine asks :meth:`last_break` where the text can stop (the
    end of a word, :attr:`TextSignal.breaks`) and :meth:`pause_at` stops it
    there. The text then ends the way a transmission ends (the carrier
    fades out; PSK31 runs half a phase reversal to zero) and, after the ID,
    begins again the way it began (an FM carrier first; the mode's own
    lead-in, :attr:`TextSignal.resume_lead_in_s`), so receivers fall back in
    step. What that adds counts in :attr:`total_samples`.
    """

    def __init__(
        self,
        signal: TextSignal,
        rate: float,
        emission: str = EMISSION_DIRECT,
        deviation_hz: float = FM_DEVIATION_HZ,
        amplitude: float = 0.9,
    ):
        if not rate > 0:
            raise ValueError("rate must be positive")
        self.signal = signal
        self.sample_rate = float(rate)
        self.emission = EMISSION_FM if emission == EMISSION_FM else EMISSION_DIRECT
        self.deviation_hz = float(deviation_hz)
        self.amplitude = float(amplitude)
        if self.emission == EMISSION_FM:
            self._lead = FM_LEAD_S
            seconds = FM_LEAD_S + signal.duration_s + FM_TAIL_S
        else:
            self._lead = 0.0
            seconds = signal.duration_s
        # The text's own samples (stops for station IDs add to them).
        self._text_total = int(round(seconds * self.sample_rate))
        self._ramp = max(1, int(round(_RISE_S * self.sample_rate)))
        self._lead_in_signal: Optional[TextSignal] = None
        self.reset()

    @property
    def duration_s(self) -> float:
        return self.total_samples / self.sample_rate

    @property
    def position(self) -> int:
        return self._pos

    @property
    def mode(self) -> str:
        """The mode the legality check knows it by."""
        return self.signal.tx_mode(self.emission)

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.signal.occupied_bandwidth_hz(self.emission, self.deviation_hz)

    def reset(self) -> None:
        """Start again from the beginning (without the stops)."""
        self._pos = 0
        self._phase = 0.0
        self.total_samples = self._text_total
        self._pieces: Deque[_Piece] = deque([_Piece(_Piece.TEXT, self._text_total)])
        self._stop_at: Optional[int] = None

    # ------------------------------------------------- stops for an ID

    def _tail_samples(self) -> int:
        return int(round(self.signal.pause_tail_s * self.sample_rate))

    def _pass_stop(self) -> None:
        """Forget a stop that has been reached (its ID was sent)."""
        if (
            self._stop_at is not None
            and self._pos >= self._stop_at
            and self._pieces
            and self._pieces[0].kind == _Piece.STOP
        ):
            self._pieces.popleft()
            self._stop_at = None

    def last_break(self, limit: int) -> Optional[int]:
        """Where a station ID can go: the last sample position (counted
        like :attr:`position`) ahead of what was read and no later than
        ``limit`` at which the text can stop for it and go on after it, the
        end of a word (:attr:`TextSignal.breaks`) further on than the last
        stop (some text always goes between two stops); None when there is
        none (or while an earlier stop is still ahead)."""
        self._pass_stop()
        breaks = self.signal._breaks
        if self._stop_at is not None or not len(breaks) or not self._pieces:
            return None
        rest = self._pieces[-1]  # (the text still to send)
        if rest.kind != _Piece.TEXT:
            return None
        text_at = np.round((breaks + self._lead) * self.sample_rate).astype(np.int64)
        offset = self.total_samples - self._text_total
        at = text_at + offset + self._tail_samples()
        ok = (
            (text_at >= rest.start + rest.done)
            # (not where the text went on after the last stop: its lead-in
            # leaves room for the fade, and it would stop there again)
            & (text_at > rest.start)
            & (text_at < rest.start + rest.length)
            # (room for the carrier to fade out before the stop)
            & (text_at + offset - self._ramp >= self._pos)
            & (at <= int(limit))
        )
        return int(at[ok][-1]) if ok.any() else None

    def pause_at(self, at: int) -> int:
        """Stop at sample ``at`` (a place :meth:`last_break` gave) for a
        station ID: :meth:`read` returns nothing past it until it is read
        again, and the text goes on after it the way it began. Returns the
        samples this adds (:attr:`total_samples` grows by as many)."""
        self._pass_stop()
        if self._stop_at is not None:
            raise ValueError("the text already stops for a station ID")
        tail = self._tail_samples()
        offset = self.total_samples - self._text_total
        text_at = int(at) - tail - offset
        rest = self._pieces[-1] if self._pieces else None
        if (
            rest is None
            or rest.kind != _Piece.TEXT
            or not rest.start + rest.done <= text_at < rest.start + rest.length
            or text_at <= rest.start  # (where it went on after the last stop)
        ):
            raise ValueError(f"the text can't stop at sample {at}")
        rate = self.sample_rate
        t = text_at / rate - self._lead  # (the text's time where it goes on)
        value = complex(self.signal.baseband(np.array([t]))[0])
        lead_in_s = self.signal.resume_lead_in_s
        align = 1 + 0j
        if lead_in_s > 0:
            if self._lead_in_signal is None:
                self._lead_in_signal = self.signal.lead_in()
            end = complex(self._lead_in_signal.baseband(np.array([lead_in_s]))[0])
            ratio = value / end if abs(end) > 1e-9 else 1 + 0j
            align = ratio / abs(ratio) if abs(ratio) > 1e-9 else 1 + 0j
        lead = int(round(FM_LEAD_S * rate)) if self.emission == EMISSION_FM else 0
        lead_in = int(round(lead_in_s * rate))
        # The text up to the stop, then the rest of it after the lead-in.
        after = _Piece(_Piece.TEXT, rest.start + rest.length - text_at, text_at)
        rest.length = text_at - rest.start
        new = [
            _Piece(_Piece.TAIL, tail, t=t, value=value),
            _Piece(_Piece.STOP, 0),
            _Piece(_Piece.LEAD, lead),
            _Piece(_Piece.LEAD_IN, lead_in, t=t, align=align),
            after,
        ]
        self._pieces.extend(p for p in new if p.length or p.kind == _Piece.STOP)
        self._stop_at = int(at)
        added = tail + lead + lead_in
        self.total_samples += added
        return added

    # ------------------------------------------------------------ reading

    def _values(self, piece: _Piece, k: int, n: int) -> np.ndarray:
        """``n`` samples of ``piece`` from its sample ``k``: the complex
        baseband (direct) or the audio (FM)."""
        rate = self.sample_rate
        if piece.kind == _Piece.TEXT:
            t = (piece.start + k + np.arange(n, dtype=np.float64)) / rate
            t -= self._lead
            if self.emission == EMISSION_DIRECT:
                return self.signal.baseband(t)
            return self.signal.audio(t)
        if piece.kind == _Piece.LEAD:
            return np.zeros(n)
        tau = (k + np.arange(n, dtype=np.float64)) / rate
        if piece.kind == _Piece.TAIL:  # (PSK31: reversals, to zero)
            tau = np.minimum(tau, self.signal.pause_tail_s)
            base = piece.value * np.cos(np.pi * PSK31_BAUD * tau)
            t = piece.t + tau
        else:  # the lead-in, in the text's time so it joins it at piece.t
            assert self._lead_in_signal is not None
            base = piece.align * self._lead_in_signal.baseband(tau)
            t = piece.t - self.signal.resume_lead_in_s + tau
        if self.emission == EMISSION_DIRECT:
            return base
        f = self.signal.settings.audio_hz
        return np.imag(base * np.exp(2j * np.pi * f * t))

    def _gain(self, start: int, n: int) -> Optional[np.ndarray]:
        """The carrier's level ramps over output samples ``start``.. (None:
        full level): it fades out before a stop and, on an FM carrier,
        rises at the start and falls at the end."""
        ramp = self._ramp
        gain: Optional[np.ndarray] = None
        pos = start + np.arange(n)
        if self.emission == EMISSION_FM and (
            start < ramp or start + n > self.total_samples - ramp
        ):
            gain = _ramp_up(pos, ramp) * _ramp_up(self.total_samples - 1 - pos, ramp)
        if self._stop_at is not None and start + n > self._stop_at - ramp:
            fade = _ramp_up(self._stop_at - 1 - pos, ramp)
            gain = fade if gain is None else gain * fade
        return gain

    def read(self, count: int) -> np.ndarray:
        """The next ``count`` samples (fewer at the end or at a stop for a
        station ID; empty when done)."""
        count = max(0, min(int(count), self.total_samples - self._pos))
        if count == 0:
            return np.zeros(0, dtype=np.complex64)
        start = self._pos
        parts: List[np.ndarray] = []
        rises: List[Tuple[int, int, int]] = []  # (where, lead sample, count)
        while count > 0 and self._pieces:
            piece = self._pieces[0]
            if piece.kind == _Piece.STOP:
                if parts:
                    break  # (the station ID goes here)
                self._pieces.popleft()
                self._stop_at = None
                continue
            n = min(count, piece.length - piece.done)
            if n > 0:
                if piece.kind == _Piece.LEAD and piece.done < self._ramp:
                    rises.append((self._pos - start, piece.done, n))
                parts.append(self._values(piece, piece.done, n))
                piece.done += n
                count -= n
                self._pos += n
            if piece.done >= piece.length:
                self._pieces.popleft()
        if not parts:
            return np.zeros(0, dtype=np.complex64)
        x = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if self.emission == EMISSION_DIRECT:
            out = x * self.amplitude
        else:
            step = 2.0 * np.pi * self.deviation_hz / self.sample_rate
            phase = self._phase + np.cumsum(x) * step
            self._phase = float(phase[-1] % (2.0 * np.pi))
            out = np.exp(1j * phase) * self.amplitude
        gain = self._gain(start, len(out))
        for where, k, n in rises:  # (the FM carrier back after a stop)
            rise = _ramp_up(k + np.arange(n), self._ramp)
            if gain is None:
                gain = np.ones(len(out))
            gain[where : where + n] *= rise
        if gain is not None:
            out = out * gain
        return out.astype(np.complex64)


#: ``report(fraction)``, 0..1; may raise to cancel (see gui/file_jobs.py).
Report = Callable[[float], None]

#: Samples generated (and written) per step when saving.
_CHUNK = 1 << 16


def write_audio_wav(
    path: str,
    signal: TextSignal,
    rate: int = 48000,
    amplitude: float = 0.5,
    report: Optional[Report] = None,
) -> int:
    """Write what a receiver plays (:meth:`TextSignal.audio`) to ``path``
    as a mono 16-bit WAV at ``rate`` (for a radio's data input or any
    player). A failed or cancelled write leaves no file. Returns the
    frames written."""
    total = signal.sample_count(rate)
    written = 0
    finished = False
    try:
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(int(rate))
            if report is not None:
                report(0.0)
            for start in range(0, total, _CHUNK):
                t = signal.times(rate, start, _CHUNK)
                audio = np.clip(signal.audio(t) * float(amplitude), -1.0, 1.0)
                out.writeframes((audio * 32767.0).astype("<i2").tobytes())
                written += len(t)
                if report is not None:
                    report(written / max(1, total))
        finished = True
    finally:
        if not finished:
            _remove(str(path))
    return written


def write_iq_file(
    path: str,
    source: TextTxSource,
    file_format: str = "sigmf",
    center_hz: float = 0.0,
    description: str = "",
    report: Optional[Report] = None,
) -> int:
    """Write the emission (``source``, read from its start) as complex
    float32 I/Q: ``file_format`` "sigmf" (``.sigmf-data`` and its
    ``.sigmf-meta``) or "cf32" (raw, with a SigMF metadata file beside it so
    the rate and frequency are kept). A failed or cancelled write leaves no
    file. Returns the samples written."""
    from ..dsp.recording import FileFormat, IQRecorder, SampleFormat, write_sidecar

    sigmf = file_format == "sigmf"
    recorder = IQRecorder(
        sample_rate=source.sample_rate,
        center_frequency=float(center_hz),
        sample_format=SampleFormat.FLOAT32,
        file_format=FileFormat.SIGMF if sigmf else FileFormat.RAW,
    )
    if description:
        recorder.set_metadata(description=description)
    source.reset()
    total = source.total_samples
    written = 0
    started = finished = False
    try:
        recorder.start(path)
        started = True
        if report is not None:
            report(0.0)
        while True:
            block = source.read(_CHUNK)
            if not len(block):
                break
            recorder.write(block)
            written += len(block)
            if report is not None:
                report(written / max(1, total))
        finished = True
    finally:
        if started:
            recorder.stop()
        if not finished:
            if sigmf:
                stem = os.path.splitext(str(path))[0]
                _remove(stem + ".sigmf-data")
                _remove(stem + ".sigmf-meta")
            else:
                _remove(str(path))
    if not sigmf:
        write_sidecar(
            path,
            SampleFormat.FLOAT32,
            source.sample_rate,
            float(center_hz),
            description=description,
        )
    return written


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

#: US rules (FCC 97.307(f), as amended in 2024, when the 300 Bd limit for
#: RTTY and data on HF was replaced by a bandwidth limit): RTTY and data
#: below 29.7 MHz may be at most 2.8 kHz wide.
HF_DATA_MAX_HZ = 29.7e6
HF_DATA_BANDWIDTH_HZ = 2800.0


def bandwidth_rule_problem(
    freq_hz: float, mode: str, bandwidth_hz: float
) -> Optional[str]:
    """Why an emission of text ``mode`` ``bandwidth_hz`` wide may not be sent
    at ``freq_hz`` under the rule above, or None. Morse code (telegraphy,
    not data) isn't limited by it."""
    if normalize_mode(mode) == MODE_CW or float(freq_hz) >= HF_DATA_MAX_HZ:
        return None
    if float(bandwidth_hz) <= HF_DATA_BANDWIDTH_HZ:
        return None
    return (
        f"the signal is {float(bandwidth_hz) / 1e3:.1f} kHz wide, and RTTY "
        "and data below 29.7 MHz may be at most 2.8 kHz wide (choose a lower "
        "baud rate or shift, or send it on the carrier)"
    )


__all__ = [
    "ASCII_BAUD",
    "ASCII_BAUDS",
    "ASCII_SHIFTS",
    "ASCII_SHIFT_HZ",
    "EMISSIONS",
    "EMISSION_DIRECT",
    "EMISSION_FM",
    "FM_DEVIATION_HZ",
    "HF_DATA_BANDWIDTH_HZ",
    "HF_DATA_MAX_HZ",
    "ITA2_FIGURES_INTL",
    "ITA2_FIGURES_US",
    "ITA2_LETTERS",
    "MAX_TEXT_CHARS",
    "MODE_ASCII",
    "MODE_CW",
    "MODE_NAMES",
    "MODE_PSK31",
    "MODE_RTTY",
    "MORSE_CODES",
    "PSK31_BAUD",
    "RTTY_BAUD",
    "RTTY_BAUDS",
    "RTTY_SHIFTS",
    "RTTY_SHIFT_HZ",
    "TEXT_MODES",
    "TextCheck",
    "TextSettings",
    "TextSignal",
    "TextTxSource",
    "VARICODE",
    "ascii_bytes",
    "audio_bounds",
    "build_text_signal",
    "char_name",
    "check_text",
    "cw_bandwidth_hz",
    "format_number",
    "ita2_codes",
    "morse_elements",
    "normalize_mode",
    "remove_unsupported",
    "bandwidth_rule_problem",
    "fsk_bandwidth_hz",
    "varicode_bits",
    "write_audio_wav",
    "write_iq_file",
]

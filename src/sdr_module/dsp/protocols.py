"""
Protocol decoders for common radio protocols.

Supports:
- POCSAG: Pager protocol (512/1200/2400 baud)
- FLEX: Motorola pager protocol (1600/3200/6400 baud)
- AX.25/APRS: Amateur packet radio
- RDS: Radio Data System (FM broadcast)
- ADS-B: Aircraft transponder (Mode S)
- ACARS: Aircraft communications
"""

import logging
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported protocol types."""

    POCSAG = "pocsag"
    FLEX = "flex"
    AX25 = "ax25"
    APRS = "aprs"
    RDS = "rds"
    ADSB = "adsb"
    ACARS = "acars"
    CW = "cw"  # Morse code (sdr_module.dsp.morse)


@dataclass
class DecodedMessage:
    """Base class for decoded messages."""

    protocol: ProtocolType
    timestamp: float
    raw_bits: bytes
    valid: bool
    error_message: str = ""


class ProtocolDecoder(ABC):
    """Abstract base class for protocol decoders."""

    def __init__(self, sample_rate: float):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self._sample_rate = sample_rate
        self._callbacks: List[Callable] = []

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def add_callback(self, callback: Callable[[DecodedMessage], None]) -> None:
        """Add callback for decoded messages."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, message: DecodedMessage) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(message)
            except (TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"Callback {callback.__name__} failed: {e}")

    @abstractmethod
    def decode(self, samples: np.ndarray) -> Sequence[DecodedMessage]:
        """Decode samples and return messages."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset decoder state."""
        pass


# ============================================================================
# Shared helpers: bit timing recovery and the BCH(31,21) code
# ============================================================================


def _as_real(samples: Any) -> np.ndarray:
    """``samples`` as a flat, finite float64 array (complex input: its real
    part; NaN and infinities become 0)."""
    x = np.asarray(samples)
    if np.iscomplexobj(x):
        x = x.real
    with np.errstate(invalid="ignore", over="ignore"):
        x = x.astype(np.float64, copy=False).ravel()
    if x.size and not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


class BitSlicer:
    """Streaming symbol timing recovery for a two-level (NRZ) baseband.

    Turns a demodulated baseband (positive = 1, negative = 0) at any sample
    rate into bits at ``baud``, keeping everything between calls: the
    fractional position of the next bit, the filter history and the last
    sample. A stream split into blocks of any size therefore gives exactly
    the bits of the whole stream.

    - Matched filter: with ``integrate`` (the default) each bit is averaged
      over one bit period (integrate-and-dump), which is what makes a noisy
      FM discriminator output usable. Leave it off for a baseband that is
      already integrated over a bit (e.g. a tone detector's output).
    - Clock recovery: a digital PLL. Before each decision, the zero
      crossing of the filtered signal (a data transition) nearest to the
      time one was expected, half a bit earlier, is compared with that
      time and the decision is moved by ``gain`` times the difference.
      Only one crossing counts per bit, so the extra crossings noise adds
      cannot drag the clock along; long runs without transitions keep the
      last timing.
    """

    def __init__(
        self,
        sample_rate: float,
        baud: float,
        integrate: bool = True,
        gain: float = 0.2,
    ):
        if not sample_rate > 0 or not baud > 0:
            raise ValueError("sample_rate and baud must be positive")
        self.sample_rate = float(sample_rate)
        self.baud = float(baud)
        self.sps = self.sample_rate / self.baud
        self._length = max(1, int(round(self.sps))) if integrate else 1
        self._gain = float(gain)
        self.reset()

    def reset(self) -> None:
        """Forget the stream."""
        self._history = np.zeros(self._length - 1, dtype=np.float64)
        # The filtered signal's last samples: a decision can fall between
        # blocks, or a little before a block after a timing correction.
        self._tail = np.zeros(int(math.ceil(self._gain * self.sps / 2.0)) + 2)
        # Position of the next decision, relative to the next block's start,
        # whether it was already adjusted (the decision itself waits for the
        # samples around it) and crossings not used yet (same reference).
        self._t = self.sps - 1.0
        self._adjusted = False
        self._carried: List[float] = []

    def process(self, samples: Any) -> List[int]:
        """Bits (0/1) decided in ``samples``."""
        return [1 if value > 0 else 0 for value in self.process_values(samples)]

    def process_values(self, samples: Any) -> List[float]:
        """The filtered signal at each decision in ``samples`` (positive:
        1), for a decoder that wants soft decisions."""
        x = _as_real(samples)
        n = len(x)
        if n == 0:
            return []
        length = self._length
        if length > 1:
            buf = np.concatenate((self._history, x))
            self._history = buf[len(buf) - (length - 1) :].copy()
            sums = np.concatenate(([0.0], np.cumsum(buf)))
            y = sums[length:] - sums[:-length]  # sum over one bit (scale-free)
        else:
            y = x
        # The filtered signal with the previous block's tail in front:
        # y[j] is yy[j + offset].
        offset = len(self._tail)
        yy = np.concatenate((self._tail, y))
        self._tail = yy[len(yy) - offset :].copy()
        # Zero crossings (fractional positions, in y's index), including one
        # between the previous block's last value and y[0].
        edge = yy[offset - 1 :]
        positive = edge > 0
        idx = np.flatnonzero(positive[1:] != positive[:-1])
        crossings = self._carried
        if len(idx):
            a, b = edge[idx], edge[idx + 1]
            crossings = crossings + (idx - 1 + a / (a - b)).tolist()

        values: List[float] = []
        t = self._t
        adjusted = self._adjusted
        sps = self.sps
        half = sps / 2.0
        gain = self._gain
        last = n - 1
        ci, nc = 0, len(crossings)
        while t <= last:
            if not adjusted:
                best = None
                while ci < nc and crossings[ci] < t:
                    err = crossings[ci] - (t - half)
                    if best is None or abs(err) < abs(best):
                        best = err
                    ci += 1
                if best is not None:
                    t += gain * best
                adjusted = True
                if t > last:
                    break
            at = t + offset
            i = int(at)
            frac = at - i
            value = yy[i] if frac == 0.0 else yy[i] + frac * (yy[i + 1] - yy[i])
            values.append(float(value))
            t += sps
            adjusted = False
        self._t = t - n
        self._adjusted = adjusted
        self._carried = [c - n for c in crossings[ci:]]
        return values


def _bch_mod(value: int, top: int) -> int:
    """``value`` (bits ``top``..0) reduced modulo the BCH(31,21) generator."""
    for i in range(top, 9, -1):
        if value >> i & 1:
            value ^= _BCH_GENERATOR << (i - 10)
    return value


# x^10 + x^9 + x^8 + x^6 + x^5 + x^3 + 1: the BCH(31,21) code of POCSAG
# and FLEX (single and double errors are correctable).
_BCH_GENERATOR = 0x769
# Syndrome of a single 1 bit at each position of a 31-bit codeword, and the
# syndrome -> error pattern table for every 1- and 2-bit error.
_BCH_BIT_SYNDROMES = [_bch_mod(1 << i, 30) for i in range(31)]
_BCH_ERRORS: Dict[int, int] = {}
for _i in range(31):
    _BCH_ERRORS.setdefault(_BCH_BIT_SYNDROMES[_i], 1 << _i)
    for _j in range(_i + 1, 31):
        _BCH_ERRORS.setdefault(
            _BCH_BIT_SYNDROMES[_i] ^ _BCH_BIT_SYNDROMES[_j], (1 << _i) | (1 << _j)
        )
# Syndromes per byte of the codeword (four table lookups per word).
_BCH_BYTE_SYNDROMES = [
    [
        _bch_mod(b << (8 * k), 30) if (b << (8 * k)) < (1 << 31) else 0
        for b in range(256)
    ]
    for k in range(4)
]


def bch3121_encode(data21: int) -> int:
    """The 31-bit BCH(31,21) codeword of ``data21`` (data in bits 30..10,
    check bits in 9..0)."""
    data21 &= 0x1FFFFF
    return (data21 << 10) | _bch_mod(data21 << 10, 30)


def bch3121_syndrome(codeword31: int) -> int:
    """Syndrome of a 31-bit codeword (0 when it is valid)."""
    c = codeword31 & 0x7FFFFFFF
    t = _BCH_BYTE_SYNDROMES
    return t[0][c & 0xFF] ^ t[1][c >> 8 & 0xFF] ^ t[2][c >> 16 & 0xFF] ^ t[3][c >> 24]


def bch3121_correct(codeword31: int) -> Tuple[bool, int, int]:
    """``(ok, corrected, errors)`` for a 31-bit codeword: up to two wrong
    bits are corrected."""
    codeword31 &= 0x7FFFFFFF
    syndrome = bch3121_syndrome(codeword31)
    if syndrome == 0:
        return True, codeword31, 0
    pattern = _BCH_ERRORS.get(syndrome)
    if pattern is None:
        return False, codeword31, 0
    return True, codeword31 ^ pattern, bin(pattern).count("1")


def _correct_word32(word: int, max_errors: int = 2) -> Tuple[bool, int]:
    """``(ok, corrected)`` for a 32-bit word: a BCH(31,21) codeword in bits
    31..1 and an even parity bit in bit 0. At most ``max_errors`` bits,
    the parity bit included, are corrected."""
    ok, fixed, errors = bch3121_correct(word >> 1)
    if not ok:
        return False, word
    parity = bin(fixed).count("1") & 1
    if parity != word & 1:
        errors += 1
    if errors > max_errors:
        return False, word
    return True, (fixed << 1) | parity


# ============================================================================
# POCSAG Protocol Decoder
# ============================================================================


@dataclass
class POCSAGMessage(DecodedMessage):
    """POCSAG decoded message."""

    address: int = 0
    function: int = 0
    message_type: str = ""  # "numeric", "alpha", "tone"
    content: str = ""
    baud_rate: int = 1200


class POCSAGDecoder(ProtocolDecoder):
    """
    POCSAG (Post Office Code Standardisation Advisory Group) decoder.

    POCSAG is a common pager protocol operating at 512, 1200, or 2400 baud.

    Input is an FM-discriminator baseband (one polarity per symbol, either
    way round) at any sample rate; consecutive ``decode()`` calls are one
    continuous stream, so blocks of any size give the same pages.

    Features:
    - Every baud rate decoded at once with ``auto_baud`` (one bit slicer
      and sync search per rate), so a mixed-rate channel decodes too
    - Clock recovery and integrate-and-dump bit decisions (see BitSlicer)
    - Either FSK polarity (found from the sync word)
    - Numeric and alphanumeric pages; tone-only pages
    - BCH(31,21) correction of up to two bit errors per codeword; a page
      with a codeword that cannot be corrected is returned as invalid
    - Pages that continue into the next batch are joined
    - Address and function code extraction
    """

    # POCSAG constants (ITU-R M.584-2)
    SYNC_WORD = 0x7CD215D8  # 32-bit frame sync per POCSAG spec
    IDLE_WORD = 0x7A89C197  # Filler codeword — pagers ignore these
    # WHY 576: spec requires ≥576 alternating bits (1010...) so all pagers
    # have time to achieve bit synchronization before the first sync word
    PREAMBLE_BITS = 576

    # WHY 0x769: BCH(31,21) generator polynomial x^10+x^9+x^8+x^6+x^5+x^3+1
    # provides single-error correction for each 31-bit codeword
    BCH_POLY = 0x769

    # Numeric character set (BCD 0-9, then spare, urgent, space, hyphen and
    # brackets); each digit is sent least significant bit first.
    NUMERIC_CHARS = "0123456789*U -)(]"

    # Alpha character set (7-bit ASCII mapping)
    ALPHA_SHIFT = {
        0x00: "\x00",
        0x01: "\x01",
        0x02: "\x02",
        0x03: "\x03",
        0x04: "\x04",
        0x05: "\x05",
        0x06: "\x06",
        0x07: "\x07",
        0x08: "\x08",
        0x09: "\x09",
        0x0A: "\n",
        0x0B: "\x0b",
        0x0C: "\x0c",
        0x0D: "\r",
        0x0E: "\x0e",
        0x0F: "\x0f",
    }

    BAUD_RATES = (512, 1200, 2400)
    # Sync word bit errors tolerated when searching (and between batches).
    _SYNC_ERRORS = 2
    _NEXT_SYNC_ERRORS = 4
    # Codewords after the first sync word of a transmission that must pass
    # the BCH check before it counts as one: noise matches the sync word
    # (within two bit errors, in one of three speeds) a few times an hour,
    # and would otherwise turn into garbage pages that failed their checks.
    _PROBATION_WORDS = 2
    # Longest page assembled (message codewords, about 570 characters)
    # before it is cut.
    _MAX_PAGE_WORDS = 200

    def __init__(
        self,
        sample_rate: float,
        baud_rate: int = 1200,
        auto_baud: bool = True,
        matched_filter: bool = True,
    ):
        """
        Initialize POCSAG decoder.

        Args:
            sample_rate: Sample rate in Hz
            baud_rate: Baud rate to decode (512, 1200, or 2400)
            auto_baud: Decode all three baud rates at once (pages report
                the rate they were sent at)
            matched_filter: Average each bit over its period before deciding
                it (leave on for a discriminator output)
        """
        super().__init__(sample_rate)
        self._baud_rate = baud_rate
        self._auto_baud = auto_baud
        self._samples_per_bit = int(sample_rate / baud_rate)
        if self._samples_per_bit < 1:
            raise ValueError(
                f"sample_rate {sample_rate:g} Hz is too low for POCSAG at "
                f"{baud_rate} baud; need at least {baud_rate} Hz"
            )
        rates = [baud_rate]
        if auto_baud:
            rates += [b for b in self.BAUD_RATES if b != baud_rate]
        self._streams = [
            _POCSAGStream(self, int(b), BitSlicer(sample_rate, b, matched_filter))
            for b in rates
            if sample_rate / b >= 2 or b == baud_rate
        ]

        # Messages
        self._messages: List[POCSAGMessage] = []
        self._timestamp = 0.0
        # Transmissions found (sync words) and pages that failed checks.
        self.frames_seen = 0
        self.frames_failed = 0

    @property
    def baud_rate(self) -> int:
        """Baud rate of the last transmission found (initially the one
        asked for)."""
        return self._baud_rate

    def _bits_to_word(self, bits: Sequence[int]) -> int:
        """Convert 32 bits to word."""
        word = 0
        for bit in bits[:32]:
            word = (word << 1) | bit
        return word

    def _check_bch(self, word: int) -> Tuple[bool, int]:
        """
        Check BCH(31,21) error correction and the parity bit.

        Up to two wrong bits (the parity bit included) are corrected.

        Returns:
            (valid, corrected_word)
        """
        return _correct_word32(word & 0xFFFFFFFF)

    def _decode_numeric(self, bits: List[int]) -> str:
        """Decode a numeric page: 4-bit characters, each sent least
        significant bit first."""
        result = ""
        for i in range(0, len(bits) - 3, 4):
            nibble = (
                bits[i] | (bits[i + 1] << 1) | (bits[i + 2] << 2) | (bits[i + 3] << 3)
            )
            result += self.NUMERIC_CHARS[nibble]
        return result.rstrip()

    def _decode_alpha(self, bits: List[int]) -> str:
        """Decode alphanumeric message."""
        result = ""
        for i in range(0, len(bits) - 6, 7):
            char_code = 0
            for j in range(7):
                char_code |= bits[i + j] << j
            if 32 <= char_code < 127:
                result += chr(char_code)
            elif char_code == 0:
                break
        return result

    def _build_message(
        self,
        batch: Sequence[int],
        address: int,
        function: int,
        message_bits: List[int],
        errors: int = 0,
        baud: Optional[int] = None,
        cut: str = "",
    ) -> POCSAGMessage:
        """Assemble a POCSAGMessage, decoding the payload per the function code.

        Function 0 selects a numeric page (4-bit BCD-style symbols); the other
        function codes select an alphanumeric page (7-bit characters). A page
        without message codewords is a tone-only page. ``errors`` counts
        codewords that could not be corrected: such a page is invalid. So is
        a page that was ``cut`` (why: it ended before its end mark).
        """
        if not message_bits:
            content = ""
            message_type = "tone"
        elif function == 0:
            content = self._decode_numeric(message_bits)
            message_type = "numeric"
        else:
            content = self._decode_alpha(message_bits)
            message_type = "alpha"
        error = ""
        if errors:
            error = (
                f"{errors} codeword{'s' if errors != 1 else ''} failed the " "BCH check"
            )
        if cut:
            error = f"{error}; {cut}" if error else cut
        return POCSAGMessage(
            protocol=ProtocolType.POCSAG,
            timestamp=self._timestamp,
            raw_bits=bytes(batch),
            valid=not (errors or cut),
            error_message=error,
            address=address,
            function=function,
            message_type=message_type,
            content=content,
            baud_rate=baud or self._baud_rate,
        )

    def _process_batch(
        self, batch: Sequence[int], flush: bool = True
    ) -> List[POCSAGMessage]:
        """Pages in one batch (16 codewords after a sync word).

        With ``flush`` a page still open at the end of the batch is
        returned too (during a live stream it may continue in the next
        batch, so ``decode`` keeps it open).
        """
        pages = _POCSAGPages(self, self._baud_rate)
        messages = pages.batch(batch)
        if flush:
            messages += pages.flush()
        return messages

    def decode(self, samples: np.ndarray) -> List[POCSAGMessage]:
        """
        Decode POCSAG from FSK-demodulated samples.

        Args:
            samples: FSK-demodulated samples (one sign per symbol; the
                polarity is found from the sync word)

        Returns:
            List of decoded POCSAG messages
        """
        x = _as_real(samples)
        self._timestamp += len(x) / self._sample_rate
        messages: List[POCSAGMessage] = []
        for stream in self._streams:
            messages += stream.feed(stream.slicer.process(x))
        for msg in messages:
            self._notify_callbacks(msg)
        self._messages = messages
        return messages

    def reset(self) -> None:
        """Reset decoder state."""
        for stream in self._streams:
            stream.reset()
        self._messages = []
        self._timestamp = 0.0


class _POCSAGPages:
    """Turns the codewords of one POCSAG stream into pages.

    A page is an address codeword followed by message codewords; it ends at
    an idle codeword, the next address codeword or the end of the
    transmission, so it may continue across batches.
    """

    def __init__(self, decoder: POCSAGDecoder, baud: int):
        self._decoder = decoder
        self.baud = baud
        self._page: Optional[Dict[str, Any]] = None
        self._orphan = False  # message codewords whose address was lost

    def reset(self) -> None:
        self._page = None
        self._orphan = False

    def batch(self, bits: Sequence[int]) -> List[POCSAGMessage]:
        """Pages completed by the 16 codewords of a batch."""
        out: List[POCSAGMessage] = []
        if len(bits) < 512:
            return out
        decoder = self._decoder
        for index in range(16):
            out += self.word(
                decoder._bits_to_word(bits[index * 32 : index * 32 + 32]), index
            )
        return out

    def word(self, word: int, index: int) -> List[POCSAGMessage]:
        """Take codeword ``index`` (0-15) of a batch; returns the page it
        completes, if any."""
        decoder = self._decoder
        ok, fixed = decoder._check_bch(word)
        if not ok:
            if self._page is not None:
                self._page["errors"] += 1
                self._page["words"].append(word)
            elif not self._orphan:
                self._orphan = True
                decoder.frames_failed += 1
            return []
        if fixed == POCSAGDecoder.IDLE_WORD:
            self._orphan = False
            return self.flush()
        if fixed & 0x80000000:
            # Message codeword: 20 data bits.
            page = self._page
            if page is None:
                if not self._orphan:
                    self._orphan = True
                    decoder.frames_failed += 1
                return []
            page["bits"].extend((fixed >> (30 - i)) & 1 for i in range(20))
            page["words"].append(fixed)
            if len(page["words"]) > POCSAGDecoder._MAX_PAGE_WORDS:
                return self.flush(
                    f"cut short after {POCSAGDecoder._MAX_PAGE_WORDS} codewords"
                )
            return []
        # Address codeword: the 18-bit address field (bits 30..13) and the
        # 2-bit function code (bits 12..11). The full 21-bit receiver
        # address is the field shifted up by 3, the frame number (0-7)
        # supplying the low 3 bits.
        out = self.flush()
        self._orphan = False
        self._page = {
            "address": (((fixed >> 13) & 0x3FFFF) << 3) | (index // 2),
            "function": (fixed >> 11) & 0x3,
            "bits": [],
            "errors": 0,
            "words": [fixed],
        }
        return out

    def flush(self, cut: str = "") -> List[POCSAGMessage]:
        """The open page, if any (the transmission or the page ended).

        ``cut`` says why a page ended before its end mark (every page ends
        at an idle or an address codeword): it may be missing the rest of
        its text, so it is returned invalid with that reason.
        """
        page, self._page = self._page, None
        if page is None:
            return []
        raw = b"".join(int(w).to_bytes(4, "big") for w in page["words"])
        msg = self._decoder._build_message(
            raw,
            page["address"],
            page["function"],
            page["bits"],
            page["errors"],
            self.baud,
            cut=cut,
        )
        if not msg.valid:
            self._decoder.frames_failed += 1
        return [msg]


class _POCSAGStream:
    """One baud rate of a POCSAG decoder: its bit slicer, the sync search
    and the batches of the transmission it is in."""

    _SEARCH, _BATCH, _NEXT_SYNC = range(3)

    def __init__(self, decoder: POCSAGDecoder, baud: int, slicer: BitSlicer):
        self._decoder = decoder
        self.baud = baud
        self.slicer = slicer
        self.pages = _POCSAGPages(decoder, baud)
        self.reset()

    def reset(self) -> None:
        self.slicer.reset()
        self.pages.reset()
        self._state = self._SEARCH
        self._reg = 0
        self._invert = 0
        self._count = 0
        self._batch: List[int] = []
        # After the first sync word of a transmission: how many codewords
        # still have to pass before it counts (0: it does), the search
        # register at that sync word and the bits received since then
        # (searched again if the sync word was noise).
        self._probation = 0
        self._sync_reg = 0
        self._held: List[int] = []

    @staticmethod
    def _word(bits: Sequence[int]) -> int:
        word = 0
        for b in bits:
            word = (word << 1) | b
        return word

    def feed(self, bits: List[int]) -> List[POCSAGMessage]:
        out: List[POCSAGMessage] = []
        sync = POCSAGDecoder.SYNC_WORD
        sync_inv = sync ^ 0xFFFFFFFF
        state, reg, invert, count = self._state, self._reg, self._invert, self._count
        batch, probation, held = self._batch, self._probation, self._held
        decoder = self._decoder
        i = 0
        while i < len(bits):
            bit = bits[i]
            i += 1
            if state == self._SEARCH:
                reg = ((reg << 1) | bit) & 0xFFFFFFFF
                if (reg ^ sync).bit_count() <= POCSAGDecoder._SYNC_ERRORS:
                    invert = 0
                elif (reg ^ sync_inv).bit_count() <= POCSAGDecoder._SYNC_ERRORS:
                    invert = 1
                else:
                    continue
                state, batch, held = self._BATCH, [], []
                probation, self._sync_reg = POCSAGDecoder._PROBATION_WORDS, reg
            elif state == self._BATCH:
                # Codewords are taken as they complete, so a page is out as
                # soon as the codeword after it arrives.
                batch.append(bit ^ invert)
                if probation:
                    held.append(bit)
                if len(batch) % 32 == 0:
                    index = len(batch) // 32 - 1
                    word = self._word(batch[-32:])
                    if not probation:
                        out += self.pages.word(word, index)
                    elif not decoder._check_bch(word)[0]:
                        # Noise that looked like a sync word: search on from
                        # the bit after it.
                        state, reg, probation = self._SEARCH, self._sync_reg, 0
                        bits, i = held + list(bits[i:]), 0
                        held, batch = [], []
                        continue
                    else:
                        probation -= 1
                        if not probation:
                            # A transmission: count it, take its codewords.
                            decoder.frames_seen += 1
                            decoder._baud_rate = self.baud
                            for k in range(index + 1):
                                word = self._word(batch[32 * k : 32 * k + 32])
                                out += self.pages.word(word, k)
                            held = []
                if len(batch) == 512:
                    state, reg, count, batch = self._NEXT_SYNC, 0, 0, []
            else:
                reg = ((reg << 1) | (bit ^ invert)) & 0xFFFFFFFF
                count += 1
                if count < 32:
                    continue
                if (reg ^ sync).bit_count() <= POCSAGDecoder._NEXT_SYNC_ERRORS:
                    state, batch = self._BATCH, []
                    continue
                # No batch follows: the transmission is over. A page still
                # open had no end mark: the rest of it was in a lost batch.
                out += self.pages.flush("cut short: the next batch was lost")
                state = self._SEARCH
                reg ^= 0xFFFFFFFF if invert else 0
        self._state, self._reg, self._invert, self._count = state, reg, invert, count
        self._batch, self._probation, self._held = batch, probation, held
        return out


# ============================================================================
# AX.25 / APRS Protocol Decoder
# ============================================================================


@dataclass
class AX25Frame(DecodedMessage):
    """AX.25 frame."""

    source: str = ""
    destination: str = ""
    digipeaters: List[str] = field(default_factory=list)
    control: int = 0
    pid: int = 0
    info: str = ""


@dataclass
class APRSMessage(DecodedMessage):
    """APRS decoded message."""

    source: str = ""
    destination: str = ""
    path: List[str] = field(default_factory=list)
    data_type: str = ""  # position, message, weather, telemetry, etc.
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    course: float = 0.0
    symbol: str = ""
    comment: str = ""
    #: What the symbol shows ("Car", "Home"...), "" when not known.
    symbol_name: str = ""
    #: Messages: whom it is for; objects and items: their name.
    addressee: str = ""


#: APRS data type (first character of the information field) -> name.
APRS_DATA_TYPES = {
    "!": "position",
    "=": "position_msg",
    "/": "position_timestamp",
    "@": "position_timestamp_msg",
    ":": "message",
    ">": "status",
    "_": "weather",
    "T": "telemetry",
    "`": "mic-e",
    "'": "mic-e",
    ";": "object",
    ")": "item",
}

#: APRS data type -> what it is, for people.
APRS_TYPE_NAMES = {
    "position": "Position",
    "position_msg": "Position (accepts messages)",
    "position_timestamp": "Position with time",
    "position_timestamp_msg": "Position with time (accepts messages)",
    "message": "Message",
    "status": "Status",
    "weather": "Weather report",
    "telemetry": "Telemetry",
    "mic-e": "Mic-E position",
    "object": "Object",
    "item": "Item",
    "unknown": "Other",
}

#: Common APRS symbols (table "/" or "\\", then the code) -> what they show.
APRS_SYMBOLS = {
    "/!": "Police station",
    "/#": "Digipeater",
    "/$": "Phone",
    "/'": "Small aircraft",
    "/-": "Home",
    "/<": "Motorcycle",
    "/=": "Train",
    "/>": "Car",
    "/O": "Balloon",
    "/P": "Police",
    "/R": "Recreational vehicle",
    "/U": "Bus",
    "/X": "Helicopter",
    "/Y": "Sailboat",
    "/[": "Person",
    "/^": "Large aircraft",
    "/_": "Weather station",
    "/a": "Ambulance",
    "/b": "Bicycle",
    "/f": "Fire truck",
    "/h": "Hospital",
    "/j": "Jeep",
    "/k": "Truck",
    "/l": "Laptop",
    "/n": "Node",
    "/r": "Repeater",
    "/s": "Boat",
    "/u": "Truck",
    "/v": "Van",
    "/y": "Yagi antenna",
    "\\#": "Digipeater",
    "\\-": "Home (HF)",
    "\\>": "Car",
    "\\_": "Weather station",
}


def _aprs_base91(text: str) -> int:
    value = 0
    for char in text:
        value = value * 91 + (ord(char) - 33)
    return value


def _aprs_position(body: str, out: Dict[str, Any]) -> str:
    """Read a position (uncompressed or compressed) at the start of
    ``body`` into ``out``; returns what follows it."""
    if len(body) >= 19 and body[:2].replace(" ", "0").isdigit():
        lat_text, table, lon_text, code = body[0:8], body[8], body[9:18], body[18]
        try:
            # Spaces are position ambiguity: read them as zeros.
            lat = int(lat_text[0:2]) + float(lat_text[2:7].replace(" ", "0")) / 60
            lon = int(lon_text[0:3]) + float(lon_text[3:8].replace(" ", "0")) / 60
        except ValueError:
            return body
        if lat_text[7] not in "NSns" or lon_text[8] not in "EWew":
            return body
        out["latitude"] = -lat if lat_text[7] in "Ss" else lat
        out["longitude"] = -lon if lon_text[8] in "Ww" else lon
        out["symbol"] = table + code
        return body[19:]
    if len(body) >= 13 and (body[0] in "/\\" or body[0].isalnum()):
        chunk = body[1:13]
        if not all(33 <= ord(c) <= 124 for c in chunk[:8]):
            return body
        out["latitude"] = 90.0 - _aprs_base91(body[1:5]) / 380926.0
        out["longitude"] = -180.0 + _aprs_base91(body[5:9]) / 190463.0
        out["symbol"] = body[0] + body[9]
        cs, kind = body[10:12], body[12]
        # Course and speed when the three bytes are base-91 (a byte that
        # isn't, e.g. one received as U+FFFD, would read as a speed of
        # 1.08 ** 65500) and the type byte doesn't say they are altitude.
        if (
            0 <= ord(cs[0]) - 33 <= 89
            and 0 <= ord(cs[1]) - 33 <= 91
            and 0 <= ord(kind) - 33 <= 91
            and (ord(kind) - 33) & 0x18 != 0x10
        ):
            out["course"] = float((ord(cs[0]) - 33) * 4)
            out["speed"] = round(1.08 ** (ord(cs[1]) - 33) - 1.0, 1)
        return body[13:]
    return body


def parse_aprs_info(info: str) -> Dict[str, Any]:
    """The APRS content of an AX.25 information field: ``data_type`` (see
    :data:`APRS_DATA_TYPES`), position (``latitude``, ``longitude``,
    uncompressed or compressed), ``symbol`` and ``symbol_name``, course
    (degrees) and speed (knots) when sent, ``altitude`` (feet, from
    "/A="), ``addressee`` (messages: to whom; objects and items: their
    name) and the ``comment`` (or the message or status text). Fields not
    sent are 0 or ""."""
    out: Dict[str, Any] = {
        "data_type": "unknown",
        "latitude": 0.0,
        "longitude": 0.0,
        "altitude": 0.0,
        "speed": 0.0,
        "course": 0.0,
        "symbol": "",
        "symbol_name": "",
        "addressee": "",
        "comment": "",
    }
    info = str(info or "")
    if not info:
        return out
    data_type = APRS_DATA_TYPES.get(info[0], "unknown")
    out["data_type"] = data_type
    body = info[1:]
    rest = body
    if data_type.startswith("position"):
        if data_type.startswith("position_timestamp"):
            body = body[7:]  # DDHHMMz, HHMMSSh or DDHHMM/
        rest = _aprs_position(body, out)
    elif data_type == "object" and len(body) >= 17:
        out["addressee"] = body[0:9].strip()
        rest = _aprs_position(body[17:], out)  # after the name, * or _, time
    elif data_type == "item":
        name, _sep, after = (
            body.partition("!") if "!" in body[:10] else body.partition("_")
        )
        out["addressee"] = name.strip()
        rest = _aprs_position(after, out)
    elif data_type == "message" and len(body) >= 10 and body[9] == ":":
        out["addressee"] = body[0:9].strip()
        rest = body[10:].split("{", 1)[0]
    elif data_type == "unknown":
        rest = info
    if out["symbol"] and data_type != "mic-e":
        # Course and speed after the symbol: "ddd/sss".
        if len(rest) >= 7 and rest[3] == "/" and (rest[:3] + rest[4:7]).isdigit():
            out["course"] = float(int(rest[:3]) % 360)
            out["speed"] = float(int(rest[4:7]))
            rest = rest[7:]
        out["symbol_name"] = APRS_SYMBOLS.get(out["symbol"], "")
    altitude = re.search(r"/A=(-?\d{5,6})", rest)
    if altitude:
        out["altitude"] = float(int(altitude.group(1)))
        rest = rest[: altitude.start()] + rest[altitude.end() :]
    out["comment"] = rest.strip()
    return out


class AX25Decoder(ProtocolDecoder):
    """
    AX.25 protocol decoder.

    AX.25 is the data link layer protocol used for packet radio,
    including APRS (Automatic Packet Reporting System).

    Input is the line signal at any sample rate: for 1200 baud packet
    (Bell 202 AFSK) a tone detector's output, positive for the mark tone
    and negative for the space tone, or the levels of direct FSK. Either
    polarity decodes (the bits are NRZI coded). Consecutive ``decode()``
    calls are one continuous stream.

    Features:
    - Clock recovery and integrate-and-dump bit decisions (see BitSlicer)
    - NRZI decoding
    - HDLC frame detection
    - Bit unstuffing
    - CRC-16 validation
    - Callsign extraction
    - APRS data parsing
    """

    FLAG = 0x7E  # HDLC frame delimiter 01111110 — six consecutive 1s are unique
    # due to bit-stuffing, so this pattern unambiguously marks frame boundaries
    CRC_POLY = 0x8408  # CRC-CCITT bit-reversed form (ITU-T V.41)
    # Shortest frame: two addresses, control and FCS; longest: AX.25's limit.
    MIN_FRAME_BYTES = 17
    MAX_FRAME_BYTES = 340

    def __init__(
        self, sample_rate: float, baud_rate: int = 1200, matched_filter: bool = True
    ):
        """
        Initialize AX.25 decoder.

        Args:
            sample_rate: Sample rate in Hz
            baud_rate: Baud rate (typically 1200 for VHF APRS)
            matched_filter: Average each bit over its period before deciding
                it (turn off for a tone detector output that already is)
        """
        super().__init__(sample_rate)
        self._baud_rate = baud_rate
        self._samples_per_bit = int(sample_rate / baud_rate)
        if self._samples_per_bit < 1:
            raise ValueError(
                f"sample_rate {sample_rate:g} Hz is too low for AX.25 at "
                f"{baud_rate} baud; need at least {baud_rate} Hz"
            )
        self._slicer = BitSlicer(sample_rate, baud_rate, matched_filter)

        # NRZI decoding state
        self._nrzi_last = 0
        # HDLC state: the last 8 bits (newest in bit 7), the octet being
        # assembled (-1 bits: not inside a frame) and the frame so far.
        self._pattern = 0
        self._octet = 0
        self._octet_bits = -1
        self._frame_bytes = bytearray()

        # Messages
        self._frames: List[AX25Frame] = []
        self._timestamp = 0.0
        # Frames found (between flags) and frames that failed the CRC.
        self.frames_seen = 0
        self.frames_failed = 0

    @property
    def baud_rate(self) -> int:
        """Get baud rate."""
        return self._baud_rate

    def _nrzi_decode(self, bit: int) -> int:
        """NRZI decode: 0 = transition, 1 = no transition."""
        if bit == self._nrzi_last:
            result = 1
        else:
            result = 0
        self._nrzi_last = bit
        return result

    def _compute_crc(self, data: bytes) -> int:
        """Compute CRC-16-CCITT."""
        crc = 0xFFFF  # WHY all-ones seed: per ITU-T CRC-16-CCITT; detects leading zeros
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ self.CRC_POLY
                else:
                    crc >>= 1
        return crc ^ 0xFFFF

    def _decode_callsign(self, data: bytes) -> Tuple[str, int]:
        """Decode AX.25 callsign from 7 bytes."""
        if len(data) < 7:
            return "", 0

        callsign = ""
        for i in range(6):
            char = (data[i] >> 1) & 0x7F
            if char != 0x20:  # Space
                callsign += chr(char)

        ssid = (data[6] >> 1) & 0x0F
        if ssid > 0:
            callsign += f"-{ssid}"

        return callsign, data[6]

    def _parse_frame(self, bits: List[int]) -> Optional[AX25Frame]:
        """Parse AX.25 frame from bits (LSB first per byte, unstuffed,
        FCS included)."""
        if len(bits) < 136:  # Minimum frame size (17 bytes)
            return None
        frame_bytes = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte |= bits[i + j] << j
            frame_bytes.append(byte)
        return self._parse_frame_bytes(bytes(frame_bytes))

    @staticmethod
    def _looks_like_ax25(data: bytes) -> bool:
        """Whether a frame that failed its CRC still looks like AX.25 (a
        destination callsign of letters, digits or spaces), so it counts
        as a damaged frame rather than noise between two flags."""
        if len(data) < 7:
            return False
        for byte in data[:6]:
            char = byte >> 1
            if byte & 1 or not (
                char == 0x20 or 0x30 <= char <= 0x39 or 0x41 <= char <= 0x5A
            ):
                return False
        return True

    def _parse_frame_bytes(self, data: bytes) -> Optional[AX25Frame]:
        """Parse an AX.25 frame (FCS included)."""
        if len(data) < self.MIN_FRAME_BYTES:
            return None

        payload = data[:-2]
        received_crc = data[-2] | (data[-1] << 8)
        computed_crc = self._compute_crc(payload)

        if received_crc != computed_crc:
            return AX25Frame(
                protocol=ProtocolType.AX25,
                timestamp=self._timestamp,
                raw_bits=bytes(data),
                valid=False,
                error_message="CRC mismatch",
            )

        # Parse addresses
        dest_call, dest_ssid = self._decode_callsign(payload[0:7])
        src_call, src_ssid = self._decode_callsign(payload[7:14])

        # Parse digipeaters
        digipeaters = []
        addr_end = 14
        if not (src_ssid & 0x01):  # More addresses follow
            while addr_end + 7 <= len(payload):
                digi_call, digi_ssid = self._decode_callsign(
                    payload[addr_end : addr_end + 7]
                )
                # "*": this digipeater has already repeated the frame.
                digipeaters.append(digi_call + ("*" if digi_ssid & 0x80 else ""))
                addr_end += 7
                if digi_ssid & 0x01:  # Last address
                    break

        if addr_end >= len(payload):
            return None

        # Control and PID
        control = payload[addr_end] if addr_end < len(payload) else 0
        pid = payload[addr_end + 1] if addr_end + 1 < len(payload) else 0

        # Info field
        info_start = addr_end + 2
        info = ""
        if info_start < len(payload):
            try:
                info = payload[info_start:].decode("ascii", errors="replace")
            except (UnicodeDecodeError, AttributeError) as e:
                logger.debug(f"Failed to decode AX.25 info field: {e}")
                info = ""

        return AX25Frame(
            protocol=ProtocolType.AX25,
            timestamp=self._timestamp,
            raw_bits=bytes(data),
            valid=True,
            source=src_call,
            destination=dest_call,
            digipeaters=digipeaters,
            control=control,
            pid=pid,
            info=info,
        )

    def parse_aprs(self, frame: AX25Frame) -> Optional[APRSMessage]:
        """
        Parse APRS data from AX.25 frame.

        Args:
            frame: AX.25 frame

        Returns:
            APRS message or None (see :func:`parse_aprs_info` for the fields)
        """
        if not frame.valid or not frame.info:
            return None
        fields = parse_aprs_info(frame.info)
        return APRSMessage(
            protocol=ProtocolType.APRS,
            timestamp=frame.timestamp,
            raw_bits=frame.raw_bits,
            valid=True,
            source=frame.source,
            destination=frame.destination,
            path=list(frame.digipeaters),
            **fields,
        )

    def decode(self, samples: np.ndarray) -> List[AX25Frame]:
        """
        Decode AX.25 frames from AFSK-demodulated samples.

        Args:
            samples: Demodulated samples (positive = mark, negative = space)

        Returns:
            List of decoded AX.25 frames: valid ones, and frames that look
            like AX.25 but failed the CRC (``valid`` False)
        """
        x = _as_real(samples)
        self._frames = []
        self._timestamp += len(x) / self._sample_rate
        for raw_bit in self._slicer.process(x):
            self._hdlc_bit(self._nrzi_decode(raw_bit))
        return self._frames

    def _hdlc_bit(self, bit: int) -> None:
        """Feed one NRZI-decoded bit to the HDLC deframer."""
        pattern = (self._pattern >> 1) | (bit << 7)
        self._pattern = pattern
        if pattern == self.FLAG:
            # The flag's first 7 bits went into the octet as data: a frame
            # that ended on a byte boundary has exactly 7 bits pending.
            if self._octet_bits == 7 and len(self._frame_bytes) >= self.MIN_FRAME_BYTES:
                self._end_frame(bytes(self._frame_bytes))
            self._octet, self._octet_bits = 0, 0
            self._frame_bytes = bytearray()
        elif pattern == 0xFE or pattern == 0xFF:
            # Seven ones: an abort, or no signal.
            self._octet_bits = -1
            self._frame_bytes = bytearray()
        elif (pattern & 0xFC) == 0x7C:
            pass  # a 0 stuffed after five 1s
        elif self._octet_bits >= 0:
            self._octet = (self._octet >> 1) | (bit << 7)
            self._octet_bits += 1
            if self._octet_bits == 8:
                self._octet_bits = 0
                if len(self._frame_bytes) >= self.MAX_FRAME_BYTES:
                    self._octet_bits = -1
                    self._frame_bytes = bytearray()
                else:
                    self._frame_bytes.append(self._octet)

    def _end_frame(self, data: bytes) -> None:
        frame = self._parse_frame_bytes(data)
        if frame is None:
            return
        if not frame.valid:
            if not self._looks_like_ax25(data):
                return  # noise between two flags
            self.frames_seen += 1
            self.frames_failed += 1
        else:
            self.frames_seen += 1
        self._frames.append(frame)
        self._notify_callbacks(frame)

    def reset(self) -> None:
        """Reset decoder state."""
        self._slicer.reset()
        self._nrzi_last = 0
        self._pattern = 0
        self._octet = 0
        self._octet_bits = -1
        self._frame_bytes = bytearray()
        self._frames = []
        self._timestamp = 0.0


# ============================================================================
# RDS (Radio Data System) Decoder
# ============================================================================


@dataclass
class RDSData(DecodedMessage):
    """RDS decoded data."""

    pi_code: int = 0  # Program Identification
    pty: int = 0  # Program Type
    tp: bool = False  # Traffic Program
    ta: bool = False  # Traffic Announcement
    ms: bool = False  # Music/Speech
    ps_name: str = ""  # Program Service name (8 chars)
    radio_text: str = ""  # Radio Text (64 chars)
    clock_time: str = ""  # CT (Clock Time)
    af_list: List[float] = field(default_factory=list)  # Alternative Frequencies
    #: Group type this came from, e.g. "0A".
    group: str = ""
    #: What changed ("pi", "ps", "radio_text", "pty", "ptyn", "clock_time"),
    #: for the updates of ``RDSDecoder.decode_bits(updates_only=True)``.
    update: str = ""
    #: The PS name has all four segments (``ps_name`` is a whole name).
    ps_complete: bool = False
    #: The RadioText has every segment up to its end.
    radio_text_complete: bool = False
    #: Name of the program type (``pty``), RBDS or RDS (see ``rbds``).
    pty_name: str = ""
    #: Program type name the station sends itself (group 10A), or "".
    ptyn: str = ""
    #: Program type names are RBDS (North America) rather than RDS.
    rbds: bool = True
    #: Extended country code (group 1A), 0 when not received.
    ecc: int = 0


def _rds_poly_mod(value: int) -> int:
    """``value`` (up to 27 bits) modulo the RDS generator polynomial
    g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1."""
    poly = 0x5B9
    for i in range(value.bit_length() - 1, 9, -1):
        if (value >> i) & 1:
            value ^= poly << (i - 10)
    return value & 0x3FF


def _build_rds_charset() -> Dict[int, str]:
    """RDS byte -> character: 0x20-0x7E read as ASCII, 0x80-0xFF from the
    RDS character table (IEC 62106 Annex E)."""
    rows = [
        "áàéèíìóòúùÑÇŞß¡Ĳ",
        "âäêëîïôöûüñçşğıĳ",
        "ªα©‰Ğěňőπ€£$←↑→↓",
        "º¹²³±İńűµ¿÷°¼½¾§",
        "ÁÀÉÈÍÌÓÒÚÙŘČŠŽÐĿ",
        "ÂÄÊËÎÏÔÖÛÜřčšžđŀ",
        "ÃÅÆŒŷÝÕØÞŊŔĆŚŹŦð",
        "ãåæœŵýõøþŋŕćśźŧ ",
    ]
    table = {code: chr(code) for code in range(0x20, 0x7F)}
    for row, chars in enumerate(rows):
        for col, char in enumerate(chars):
            table[0x80 + 16 * row + col] = char
    return table


_RDS_CHARSET = _build_rds_charset()


def rds_char(code: int) -> str:
    """The character an RDS byte stands for ("" for codes that print
    nothing; unknown codes as a space)."""
    if code in (0x0A, 0x0B):  # line break, end of headline
        return " "
    if code == 0x1F:  # soft hyphen
        return ""
    return _RDS_CHARSET.get(code, " ")


def _mjd_date(mjd: int) -> Optional[Tuple[int, int, int]]:
    """(year, month, day) of a Modified Julian Day (IEC 62106 Annex G)."""
    if mjd <= 15079:
        return None
    yp = int((mjd - 15078.2) / 365.25)
    mp = int((mjd - 14956.1 - int(yp * 365.25)) / 30.6001)
    day = mjd - 14956 - int(yp * 365.25) - int(mp * 30.6001)
    k = 1 if mp in (14, 15) else 0
    year = 1900 + yp + k
    month = mp - 1 - k * 12
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year, month, day


class RDSDecoder(ProtocolDecoder):
    """
    RDS (Radio Data System) decoder: groups from the RDS bit stream.

    RDS is broadcast on FM radio on a 57 kHz subcarrier at 1187.5 bit/s
    (:class:`sdr_module.dsp.rds.RDSDemodulator` turns the FM composite into
    those bits). Every group is four 26-bit blocks: 16 data bits and a
    10-bit check word with an offset word (A, B, C or C', D) added, which
    tells the block's place in the group.

    Two ways in, both streaming (the bits carry over between calls):

    - :meth:`decode_bits`: the data bits (after biphase and differential
      decoding);
    - :meth:`decode`: a baseband of NRZ bit levels (positive: 1) at
      ``sample_rate``, sliced at 1187.5 bit/s with clock recovery.

    Features:
    - Block synchronization: until in sync, a block is recognised only by
      an exact check word; two blocks in a row, 26 bits apart and in order,
      give sync. In sync, every block is checked against the offset word
      expected next, and error bursts of up to ``MAX_BURST`` bits are
      corrected (one wrong bit on the air becomes two after differential
      decoding). Sync is lost after ``SYNC_LOSS_BLOCKS`` bad blocks in a
      row, and found again at once if the stream slipped a bit.
    - Error detection and correction (see above)
    - PI code extraction (block A, or block C' of version B groups)
    - PS (Program Service) name decoding (groups 0A/0B), whole once all
      four segments arrived
    - RT (Radio Text) decoding (groups 2A/2B), whole once every segment up
      to its end arrived
    - Clock time decoding (group 4A), PTY (every group), PTYN (10A), TA,
      MS, TP, alternative frequencies (0A) and the extended country code
      (1A), which picks RDS or RBDS program type names
    """

    # RDS constants
    BLOCK_SIZE = 26  # bits
    GROUP_SIZE = 4  # blocks
    BAUD_RATE = 1187.5

    # Syndrome values for block types. For an error-free block the syndrome
    # (the 26-bit block reduced modulo the generator polynomial) equals the
    # block's offset word, because the transmitted check bits carry
    # ``(message * x^10 mod g) XOR offset``. These are therefore the standard
    # IEC 62106 offset words A, B, C, C' and D.
    SYNDROMES = {
        0x0FC: "A",
        0x198: "B",
        0x168: "C",
        0x350: "C'",
        0x1B4: "D",
    }

    # Offset words
    OFFSETS = {
        "A": 0x0FC,
        "B": 0x198,
        "C": 0x168,
        "C'": 0x350,
        "D": 0x1B4,
    }

    # Place of each block type in a group.
    _SLOT = {"A": 0, "B": 1, "C": 2, "C'": 2, "D": 3}

    #: Longest burst of wrong bits corrected in a block (while in sync).
    MAX_BURST = 2
    #: Bad blocks in a row that lose sync.
    SYNC_LOSS_BLOCKS = 8

    # PTY codes (North America)
    PTY_CODES = [
        "None",
        "News",
        "Information",
        "Sports",
        "Talk",
        "Rock",
        "Classic Rock",
        "Adult Hits",
        "Soft Rock",
        "Top 40",
        "Country",
        "Oldies",
        "Soft",
        "Nostalgia",
        "Jazz",
        "Classical",
        "R&B",
        "Soft R&B",
        "Language",
        "Religious Music",
        "Religious Talk",
        "Personality",
        "Public",
        "College",
        "Spanish Talk",
        "Spanish Music",
        "Hip Hop",
        "Unassigned",
        "Unassigned",
        "Weather",
        "Emergency Test",
        "Emergency",
    ]

    # PTY codes (RDS, outside North America)
    PTY_CODES_RDS = [
        "None",
        "News",
        "Current Affairs",
        "Information",
        "Sport",
        "Education",
        "Drama",
        "Culture",
        "Science",
        "Varied",
        "Pop Music",
        "Rock Music",
        "Easy Listening",
        "Light Classical",
        "Serious Classical",
        "Other Music",
        "Weather",
        "Finance",
        "Children's Programmes",
        "Social Affairs",
        "Religion",
        "Phone-in",
        "Travel",
        "Leisure",
        "Jazz Music",
        "Country Music",
        "National Music",
        "Oldies Music",
        "Folk Music",
        "Documentary",
        "Alarm Test",
        "Alarm",
    ]

    # Extended country codes of the countries that use RBDS program types.
    RBDS_ECC = frozenset({0xA0, 0xA1, 0xA5})  # USA, Canada, Mexico

    _BURSTS: Dict[int, int] = {}  # syndrome of an error burst -> the burst

    def __init__(self, sample_rate: float):
        """
        Initialize RDS decoder.

        Args:
            sample_rate: Sample rate in Hz of the baseband given to
                :meth:`decode` (RDS is 1187.5 baud)
        """
        super().__init__(sample_rate)
        self._baud_rate = self.BAUD_RATE
        self._samples_per_bit = sample_rate / self._baud_rate
        self._slicer = BitSlicer(sample_rate, self._baud_rate)
        if not RDSDecoder._BURSTS:
            RDSDecoder._BURSTS = self._burst_table(5)
        # x^26 mod g: the bit leaving the 26-bit window, in the syndrome.
        self._x26 = _rds_poly_mod(1 << 26)
        self._timestamp = 0.0
        # Counters for the live decoder's status: groups found in sync,
        # groups whose block B (the group type) could not be read, and
        # blocks received good, corrected and failed while in sync.
        self.frames_seen = 0
        self.frames_failed = 0
        self.blocks_good = 0
        self.blocks_corrected = 0
        self.blocks_failed = 0
        # The bit stream: bits so far, the last 26 bits and their syndrome,
        # the blocks that checked exactly in the last 26 positions, the sync
        # and the group being received ((data, block type, corrected) per
        # block; partial: it began before the sync).
        self._bit_index = 0
        self._reg = 0
        self._syn = 0
        self._hits: List[Optional[Tuple[str, int]]] = []
        self._synced = False
        self._in_block = 0
        self._bad_run = 0
        self._expect = 0
        self._partial = False
        self._group: List[Optional[Tuple[int, str, bool]]] = []
        self._snapshots: List[RDSData] = []
        self._updates: List[RDSData] = []
        # The station: codes (PI and PTY change when seen twice in a row),
        # name, RadioText, clock, program type name, frequencies, country.
        self._pi_code = 0
        self._pty = 0
        self._tp = False
        self._ta = False
        self._ms = False
        self._known: Dict[str, bool] = {}
        self._pending: Dict[str, Optional[int]] = {}
        self._ps: List[str] = []
        self._ps_seen = 0
        self._ps_done = ""
        self._rt: List[str] = []
        self._rt_seen = 0
        self._rt_flag: Optional[int] = None
        self._rt_version: Optional[int] = None
        self._rt_end: Optional[int] = None
        self._rt_done = ""
        self._ct = ""
        self._ptyn: List[str] = []
        self._ptyn_seen = 0
        self._ptyn_flag: Optional[int] = None
        self._ptyn_done = ""
        self._text_fixed = False  # the group's text blocks were corrected
        self._af: Set[float] = set()
        self._ecc = 0
        self.reset()

    # -- the code ------------------------------------------------------------

    @classmethod
    def _burst_table(cls, longest: int) -> Dict[int, int]:
        """Syndrome -> error pattern for every burst of 1..``longest`` bits
        (the code tells them all apart up to 5 bits); shorter bursts first."""
        table: Dict[int, int] = {}
        for length in range(1, longest + 1):
            if length == 1:
                patterns = [1]
            else:
                inner = length - 2
                patterns = [
                    (1 << (length - 1)) | (middle << 1) | 1
                    for middle in range(1 << inner)
                ]
            for pattern in patterns:
                for shift in range(cls.BLOCK_SIZE - length + 1):
                    error = pattern << shift
                    table.setdefault(_rds_poly_mod(error), error)
        return table

    def _syndrome(self, block: int) -> int:
        """Return the RDS block syndrome: the 26-bit block reduced modulo the
        generator polynomial g(x) = x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1.

        The check bits of a block carry ``(message * x^10 mod g) XOR offset``,
        so an error-free block reduces to its offset word; ``_decode_block``
        matches that against ``SYNDROMES`` to identify the block type. (The
        previous LFSR fed the feedback bit back into the register and XORed an
        11-bit polynomial into a 10-bit register, so it never produced any of
        the tabulated values and the decoder recognised nothing.)
        """
        poly = 0x5B9  # includes the x^10 term (degree 10)
        reg = block & 0x3FFFFFF  # 26-bit block
        for i in range(25, 9, -1):
            if (reg >> i) & 1:
                reg ^= poly << (i - 10)
        return reg & 0x3FF

    def _decode_block(self, bits: List[int]) -> Tuple[int, str]:
        """
        Decode 26-bit block.

        Returns:
            (16-bit data, block_type) or (0, '') if invalid
        """
        if len(bits) != 26:
            return 0, ""

        block = 0
        for bit in bits:
            block = (block << 1) | bit

        # Calculate syndrome
        syn = self._syndrome(block)

        # Check against known syndromes
        for expected_syn, block_type in self.SYNDROMES.items():
            if syn == expected_syn:
                # Extract 16-bit data
                data = (block >> 10) & 0xFFFF
                return data, block_type

        return 0, ""

    def _correct(
        self, block: int, syndrome: int, kinds: Sequence[str]
    ) -> Optional[Tuple[int, str, bool]]:
        """``(data, kind, corrected)`` of a block expected to be one of
        ``kinds``: exact, or with a short error burst corrected; None when
        it cannot be read."""
        for kind in kinds:
            if syndrome == self.OFFSETS[kind]:
                return (block >> 10) & 0xFFFF, kind, False
        other = self.SYNDROMES.get(syndrome)
        if other is not None and all(self._SLOT[other] != self._SLOT[k] for k in kinds):
            # It checks exactly as another block of the group: a block out
            # of place (whole blocks were lost), not a damaged one. (Offset
            # words A, B, C' and D are only a one- or two-bit burst from
            # their neighbours', so "correcting" it would make, e.g., the
            # PI code of block A into a block B of garbage.)
            return None
        for kind in kinds:
            error = self._BURSTS.get(syndrome ^ self.OFFSETS[kind])
            if error is not None and self._burst_length(error) <= self.MAX_BURST:
                return ((block ^ error) >> 10) & 0xFFFF, kind, True
        return None

    @staticmethod
    def _burst_length(error: int) -> int:
        low = (error & -error).bit_length()
        return error.bit_length() - low + 1

    # -- the stream ----------------------------------------------------------

    @property
    def synced(self) -> bool:
        """Whether the decoder is in block sync."""
        return self._synced

    @property
    def pi_known(self) -> bool:
        """Whether the station's PI code has been received."""
        return self._known["pi"]

    def decode(self, samples: np.ndarray) -> List[RDSData]:
        """
        Decode RDS from a baseband of NRZ bit levels.

        Args:
            samples: Bit levels (positive = 1) at ``sample_rate``, e.g. a
                BPSK-demodulated, biphase- and differentially decoded RDS
                signal resampled to bit levels

        Returns:
            The state after every group decoded in these samples (see
            :meth:`decode_bits`)
        """
        x = _as_real(samples)
        self._timestamp += len(x) / self._sample_rate
        return self.decode_bits(self._slicer.process(x))

    def decode_bits(
        self, bits: Sequence[int], updates_only: bool = False
    ) -> List[RDSData]:
        """
        Decode RDS data bits (0/1), any number per call.

        Args:
            bits: The data bits, after biphase and differential decoding
            updates_only: Return only what changed (one :class:`RDSData`
                per change, its ``update`` naming it: the station's PI code,
                a whole new PS name or RadioText, the program type, the
                program type name or the clock time) instead of the state
                after every group

        Returns:
            :class:`RDSData` snapshots (see ``updates_only``)
        """
        self._snapshots = []
        self._updates = []
        reg, syn = self._reg, self._syn
        x26 = self._x26
        hits = self._hits
        for bit in bits:
            bit = 1 if bit else 0
            index = self._bit_index
            self._bit_index = index + 1
            leaving = (reg >> 25) & 1
            reg = ((reg << 1) | bit) & 0x3FFFFFF
            syn = (syn << 1) | bit
            if syn & 0x400:
                syn ^= 0x5B9
            if leaving:
                syn ^= x26
            # A block that checks exactly, ending at this bit.
            kind = self.SYNDROMES.get(syn) if index >= 25 else None
            slot = index % self.BLOCK_SIZE
            previous = hits[slot]  # what ended 26 bits ago
            hits[slot] = (kind, reg) if kind else None
            if self._synced:
                self._in_block += 1
                if self._in_block == self.BLOCK_SIZE:
                    self._in_block = 0
                    if (
                        kind
                        and previous
                        and self._SLOT[kind] != self._expect
                        and self._SLOT[previous[0]] == (self._SLOT[kind] - 1) % 4
                    ):
                        # Two good blocks in a row, in order, but not the
                        # ones expected here: whole blocks were lost (a
                        # dropout). Sync on them.
                        self._acquire(previous, kind, reg)
                    else:
                        self._synced_block(reg, syn)
                elif (
                    kind
                    and previous
                    and self._bad_run >= 2
                    and self._SLOT[previous[0]] == (self._SLOT[kind] - 1) % 4
                ):
                    # The stream slipped: two good blocks in a row at
                    # another place. Sync there.
                    self._acquire(previous, kind, reg)
            elif (
                kind
                and previous
                and self._SLOT[previous[0]] == (self._SLOT[kind] - 1) % 4
            ):
                self._acquire(previous, kind, reg)
        self._reg, self._syn = reg, syn
        return list(self._updates if updates_only else self._snapshots)

    def _acquire(self, previous: Tuple[str, int], kind: str, reg: int) -> None:
        """Block sync from two good blocks in a row."""
        self._synced = True
        self._in_block = 0
        self._bad_run = 0
        self._group = [None] * self.GROUP_SIZE
        prev_kind, prev_reg = previous
        prev_slot, slot = self._SLOT[prev_kind], self._SLOT[kind]
        if prev_slot < slot:  # the same group
            self._group[prev_slot] = ((prev_reg >> 10) & 0xFFFF, prev_kind, False)
        self._group[slot] = ((reg >> 10) & 0xFFFF, kind, False)
        self.blocks_good += 1
        self._expect = (slot + 1) % self.GROUP_SIZE
        # Blocks of this group before the sync were not seen: a missing B
        # there is no failure.
        self._partial = self._group[1] is None
        if slot == self.GROUP_SIZE - 1:
            self._group_done()

    def _synced_block(self, reg: int, syn: int) -> None:
        """The next block, in sync."""
        slot = self._expect
        if slot == 2:
            version = self._group_version()
            kinds: Tuple[str, ...] = (
                ("C",) if version == 0 else ("C'",) if version == 1 else ("C", "C'")
            )
        else:
            kinds = (("A", "B", "", "D")[slot],)
        read = self._correct(reg, syn, kinds)
        if read is None:
            self.blocks_failed += 1
            self._bad_run += 1
            self._group[slot] = None
        else:
            data, kind, corrected = read
            if corrected:
                self.blocks_corrected += 1
            else:
                self.blocks_good += 1
            self._bad_run = 0
            self._group[slot] = (data, kind, corrected)
        self._expect = (slot + 1) % self.GROUP_SIZE
        if slot == self.GROUP_SIZE - 1:
            self._group_done()
        if self._bad_run >= self.SYNC_LOSS_BLOCKS:
            self._synced = False
            self._group = [None] * self.GROUP_SIZE

    def _group_version(self) -> Optional[int]:
        block_b = self._group[1]
        return None if block_b is None else (block_b[0] >> 11) & 1

    def _group_done(self) -> None:
        group = self._group
        self._group = [None] * self.GROUP_SIZE
        partial, self._partial = self._partial, False
        if group[1] is None:
            if not partial:
                self.frames_seen += 1
                self.frames_failed += 1  # no group type: nothing to read
            return
        self.frames_seen += 1
        rds = self._process_blocks(group)
        if rds is not None:
            self._snapshots.append(rds)
            self._notify_callbacks(rds)

    # -- groups --------------------------------------------------------------

    def _process_blocks(
        self, group: Sequence[Optional[Tuple[int, str, bool]]]
    ) -> Optional[RDSData]:
        """Read a group (blocks A to D as (data, type, corrected); None for
        one that could not be read); returns the decoder's state after it.

        A block whose errors were corrected may fill in text not received
        yet, but not change text already received: a wrong correction (rare,
        but possible in noise) does not garble the station name."""
        block_b = group[1]
        if block_b is None:
            return None
        data_b = block_b[0]
        group_type = (data_b >> 12) & 0x0F
        version = (data_b >> 11) & 0x01
        name = f"{group_type}{'AB'[version]}"
        c = group[2][0] if group[2] is not None else None
        d = group[3][0] if group[3] is not None else None
        self._text_fixed = any(
            block is not None and block[2] for block in (group[2], group[3])
        )

        # PI: block A, or block C' of a version B group.
        pi_code = None
        if group[0] is not None:
            pi_code = group[0][0]
        elif group[2] is not None and group[2][1] == "C'":
            pi_code = group[2][0]
        if pi_code is not None:
            self._confirm("pi", pi_code)

        self._tp = bool((data_b >> 10) & 0x01)
        self._confirm("pty", (data_b >> 5) & 0x1F)

        if group_type == 0:
            self._group_0(data_b, version, c, d)
        elif group_type == 1 and version == 0 and c is not None:
            if (c >> 12) & 0x07 == 0:  # variant 0: the ECC
                self._ecc = c & 0xFF
        elif group_type == 2:
            self._group_2(data_b, version, c, d)
        elif group_type == 4 and version == 0 and c is not None and d is not None:
            self._group_4a(data_b, c, d)
        elif group_type == 10 and version == 0 and c is not None and d is not None:
            self._group_10a(data_b, c, d)
        return self._snapshot(name)

    def _confirm(self, what: str, value: int) -> None:
        """PI and PTY change only when the new value comes twice in a row
        (a wrongly corrected block does not change the station)."""
        current = self._pi_code if what == "pi" else self._pty
        if value == current and self._known[what]:
            self._pending[what] = None
            return
        if self._pending.get(what) == value or not self._known[what]:
            if what == "pi":
                self._pi_code = value
            else:
                self._pty = value
            self._pending[what] = None
            self._known[what] = True
            self._changed(what)
        else:
            self._pending[what] = value

    def _group_0(
        self, data_b: int, version: int, c: Optional[int], d: Optional[int]
    ) -> None:
        self._ta = bool((data_b >> 4) & 0x01)
        self._ms = bool((data_b >> 3) & 0x01)
        if version == 0 and c is not None:
            for code in ((c >> 8) & 0xFF, c & 0xFF):
                if 1 <= code <= 204:
                    self._af.add(round(87.5 + 0.1 * code, 1))
        if d is None:
            return
        segment = data_b & 0x03
        chars = rds_char((d >> 8) & 0xFF) or " ", rds_char(d & 0xFF) or " "
        at = 2 * segment
        if (self._ps_seen >> segment) & 1 and (self._ps[at], self._ps[at + 1]) != chars:
            if self._text_fixed:
                return  # a corrected block does not change a name
            # A new name is coming: the other segments are the old one's.
            self._ps_seen = 0
        self._ps[at], self._ps[at + 1] = chars
        self._ps_seen |= 1 << segment
        if self._ps_seen == 0x0F:
            name = "".join(self._ps)
            if name != self._ps_done:
                self._ps_done = name
                self._changed("ps")

    def _group_2(
        self, data_b: int, version: int, c: Optional[int], d: Optional[int]
    ) -> None:
        flag = (data_b >> 4) & 0x01
        segment = data_b & 0x0F
        if flag != self._rt_flag or version != self._rt_version:
            # A new text (the A/B flag toggled): start afresh.
            self._rt_flag = flag
            self._rt_version = version
            self._rt = [" "] * 64
            self._rt_seen = 0
            self._rt_end = None
        if version == 0:
            if c is None or d is None:
                return
            codes = [(c >> 8) & 0xFF, c & 0xFF, (d >> 8) & 0xFF, d & 0xFF]
            width = 4
        else:
            if d is None:
                return
            codes = [(d >> 8) & 0xFF, d & 0xFF]
            width = 2
        at = segment * width
        chars: List[str] = []
        end = None
        for i, code in enumerate(codes):
            if code == 0x0D:  # end of the text
                end = at + i
                break
            chars.append(rds_char(code) or " ")
        if (self._rt_seen >> segment) & 1 and (
            self._rt[at : at + len(chars)] != chars
            or (end is not None and end != self._rt_end)
        ):
            if self._text_fixed:
                return  # a corrected block does not change the text
            # The text changed without the A/B flag: start it afresh.
            self._rt_seen = 0
            self._rt_end = None
        self._rt[at : at + len(chars)] = chars
        if end is not None:
            self._rt_end = end if self._rt_end is None else min(self._rt_end, end)
        self._rt_seen |= 1 << segment
        length = self._rt_end if self._rt_end is not None else 16 * width
        needed = (1 << (-(-length // width))) - 1 if length else 1
        if self._rt_seen & needed == needed:
            text = "".join(self._rt[:length]).rstrip()
            if text and text != self._rt_done:
                self._rt_done = text
                self._changed("radio_text")

    def _group_4a(self, data_b: int, c: int, d: int) -> None:
        mjd = ((data_b & 0x03) << 15) | (c >> 1)
        hour = ((c & 0x01) << 4) | (d >> 12)
        minute = (d >> 6) & 0x3F
        offset = (d & 0x1F) * (-1 if (d >> 5) & 0x01 else 1)  # half hours
        date = _mjd_date(mjd)
        if date is None or hour > 23 or minute > 59 or abs(offset) > 28:
            return
        from datetime import datetime, timedelta

        try:
            utc = datetime(*date, hour, minute)
        except ValueError:
            return
        local = utc + timedelta(minutes=30 * offset)
        sign = "-" if offset < 0 else "+"
        minutes = abs(offset) * 30
        text = (
            f"{local:%Y-%m-%d %H:%M} (UTC{sign}{minutes // 60:02d}:{minutes % 60:02d})"
        )
        if text != self._ct:
            self._ct = text
            self._changed("clock_time")

    def _group_10a(self, data_b: int, c: int, d: int) -> None:
        flag = (data_b >> 4) & 0x01
        segment = data_b & 0x01
        if flag != self._ptyn_flag:
            self._ptyn_flag = flag
            self._ptyn = [" "] * 8
            self._ptyn_seen = 0
        codes = [(c >> 8) & 0xFF, c & 0xFF, (d >> 8) & 0xFF, d & 0xFF]
        for i, code in enumerate(codes):
            self._ptyn[4 * segment + i] = rds_char(code) or " "
        self._ptyn_seen |= 1 << segment
        if self._ptyn_seen == 0x03:
            name = "".join(self._ptyn).strip()
            if name != self._ptyn_done:
                self._ptyn_done = name
                self._changed("ptyn")

    # -- state ---------------------------------------------------------------

    @property
    def rbds(self) -> bool:
        """Whether program types are named as RBDS (North America): yes
        unless the station's extended country code says otherwise."""
        return not self._ecc or self._ecc in self.RBDS_ECC

    def get_pty_name(self, pty: int, rbds: Optional[bool] = None) -> str:
        """Get PTY name from code (RBDS names unless ``rbds`` is False)."""
        names = self.PTY_CODES if rbds is None or rbds else self.PTY_CODES_RDS
        if 0 <= pty < len(names):
            return names[pty]
        return "Unknown"

    def _changed(self, what: str) -> None:
        self._updates.append(self._snapshot("", update=what))

    def _snapshot(self, group: str, update: str = "") -> RDSData:
        rbds = self.rbds
        return RDSData(
            protocol=ProtocolType.RDS,
            timestamp=self._timestamp,
            raw_bits=bytes(),
            valid=True,
            pi_code=self._pi_code,
            pty=self._pty,
            tp=self._tp,
            ta=self._ta,
            ms=self._ms,
            ps_name=(self._ps_done if self._ps_done else "".join(self._ps)).strip(),
            radio_text=(self._rt_done or "".join(self._rt).rstrip()).strip(),
            clock_time=self._ct,
            af_list=sorted(self._af),
            group=group,
            update=update,
            ps_complete=bool(self._ps_done),
            radio_text_complete=bool(self._rt_done),
            pty_name=self.get_pty_name(self._pty, rbds),
            ptyn=self._ptyn_done,
            rbds=rbds,
            ecc=self._ecc,
        )

    def station(self) -> RDSData:
        """What is known about the station now."""
        return self._snapshot("")

    def reset(self) -> None:
        """Reset decoder state (the bit stream and the station's data)."""
        self._slicer.reset()
        self._bit_index = 0
        self._reg = 0
        self._syn = 0
        self._hits = [None] * self.BLOCK_SIZE
        self._synced = False
        self._in_block = 0
        self._bad_run = 0
        self._expect = 0
        self._partial = False
        self._group = [None] * self.GROUP_SIZE
        self._snapshots = []
        self._updates = []
        self._pi_code = 0
        self._pty = 0
        self._tp = False
        self._ta = False
        self._ms = False
        self._known = {"pi": False, "pty": False}
        self._pending = {"pi": None, "pty": None}
        self._ps = [" "] * 8
        self._ps_seen = 0
        self._ps_done = ""
        self._rt = [" "] * 64
        self._rt_seen = 0
        self._rt_flag = None
        self._rt_version = None
        self._rt_end = None
        self._rt_done = ""
        self._ct = ""
        self._ptyn = [" "] * 8
        self._ptyn_seen = 0
        self._ptyn_flag = None
        self._ptyn_done = ""
        self._text_fixed = False
        self._af = set()
        self._ecc = 0
        self._timestamp = 0.0


# ============================================================================
# ADS-B (Automatic Dependent Surveillance-Broadcast) Decoder
# ============================================================================


@dataclass
class ADSBMessage(DecodedMessage):
    """ADS-B decoded message."""

    icao_address: str = ""  # 24-bit ICAO aircraft address
    downlink_format: int = 0  # DF (17 for ADS-B)
    type_code: int = 0  # TC (message type)
    callsign: str = ""  # Aircraft callsign (8 chars)
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: int = 0  # Altitude in feet
    velocity: float = 0.0  # Ground speed in knots
    heading: float = 0.0  # Track angle in degrees
    vertical_rate: int = 0  # Vertical rate in ft/min
    squawk: str = ""  # Transponder code
    on_ground: bool = False
    category: str = ""  # Aircraft category
    #: The velocity message carried a vertical rate (an all-zero field
    #: means "no information").
    vertical_rate_known: bool = True


class ADSBDecoder(ProtocolDecoder):
    """
    ADS-B (Mode S) decoder.

    Decodes the Mode S replies and ADS-B extended squitters that aircraft
    transponders send on 1090 MHz. Validation follows dump1090, because a
    preamble-like burst turns up in receiver noise many times a second and
    every one of them carries some "address":

    - DF17/18 (extended squitter) must have a CRC residual of 0.
    - DF11 (all-call reply) may carry an interrogator code (IID) in the low
      7 bits of the residual; the other 17 bits must be 0. A reply with a
      non-zero IID is only accepted from an aircraft already known.
    - DF0/4/5/16/20/21 (address/parity) carry the aircraft address XORed
      into the parity, so the address is recovered as CRC XOR parity. Such a
      reply is accepted only if that address was seen in a CRC-checked
      DF17/18 (or a DF11 with IID 0) within ``ICAO_TIMEOUT_S`` seconds of
      signal time.
    - Anything else (other formats, failed checks) is dropped, not returned.

    Features:
    - Mode S preamble detection (dump1090's pulse/quiet test) on the
      magnitude signal in 0.5 us slots. Except at exactly 2 MS/s the slots
      are read at 4 or 5 phases (every 0.1-0.125 us), so a reply decodes
      wherever it starts between two samples; the best-aligned phase of
      each reply is demodulated
    - Frames that straddle two ``decode()`` calls are still found
    - Aircraft identification (callsign)
    - Airborne position decoding (CPR)
    - Velocity decoding
    - Altitude (DF17 position messages and the AC13 field of DF0/4/16/20)
    - Squawk (identity field of DF5/21)
    """

    # Mode S constants
    PREAMBLE_US = 8.0  # Preamble duration in microseconds
    SHORT_MSG_BITS = 56  # Short message (DF 0, 4, 5, 11)
    LONG_MSG_BITS = 112  # Long message (DF 16, 17, 18, 20, 21)

    # CRC-24 polynomial (0x1FFF409)
    CRC_POLY = 0x1FFF409
    CRC_TABLE: List[int] = []

    # Character lookup for callsign
    CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

    # Emitter category names by identification type code (sets D, C, B, A)
    # and category number (DO-260B); "" for none or reserved.
    CATEGORIES = {
        1: ("",) * 8,
        2: (
            "",
            "Emergency vehicle",
            "Service vehicle",
            "Ground obstruction",
            "Ground obstruction",
            "Ground obstruction",
            "Ground obstruction",
            "Ground obstruction",
        ),
        3: (
            "",
            "Glider",
            "Lighter than air",
            "Parachutist",
            "Ultralight",
            "",
            "Unmanned aircraft",
            "Space vehicle",
        ),
        4: (
            "",
            "Light",
            "Small",
            "Large",
            "High vortex large",
            "Heavy",
            "High performance",
            "Rotorcraft",
        ),
    }

    # The decoder works on 0.5 us slots: 2 per bit, 16 per preamble.
    WORK_RATE = 2e6
    # How long an aircraft address stays "known" for address/parity replies.
    ICAO_TIMEOUT_S = 60.0

    _PREAMBLE_SLOTS = 16  # 8 us
    _FRAME_SLOTS = 16 + 2 * 112  # preamble + long message
    _SHORT_FORMATS = (0, 4, 5, 11)
    _LONG_FORMATS = (16, 17, 18, 20, 21)
    _BATCH = 4096  # candidates demodulated per numpy batch

    def __init__(self, sample_rate: float):
        """
        Initialize ADS-B decoder.

        Args:
            sample_rate: Sample rate in Hz (at least 2 MHz; the magnitude is
                read in 0.5 us slots at any rate, see ``_plan_slots``)
        """
        super().__init__(sample_rate)

        # 1090 MHz Mode S: 1 us per bit, sent as a pulse in one of its two
        # 0.5 us halves. Below 2 MHz the halves cannot be told apart, so
        # reject unusable rates up front.
        if sample_rate < 2e6:
            raise ValueError(
                f"ADS-B (Mode S) requires a sample rate of at least 2 MHz, "
                f"got {sample_rate:g} Hz"
            )
        self._ratio = sample_rate / self.WORK_RATE
        self._plan_slots()

        # Generate CRC table
        if not ADSBDecoder.CRC_TABLE:
            ADSBDecoder.CRC_TABLE = self._generate_crc_table()

        # State
        self._messages: List[ADSBMessage] = []
        self._timestamp = 0.0
        # Input samples not yet turned into slots (they start a period of
        # the slot pattern, see _plan_slots).
        self._rest = np.zeros(0, dtype=np.float32)
        # Unscanned end of the previous block (slot values) and how many of
        # its positions belong to a frame that was already decoded.
        self._tail = np.zeros(0, dtype=np.float32)
        self._skip = 0
        # Addresses from CRC-checked replies: icao (int) -> signal time seen.
        self._known_icao: Dict[int, float] = {}

        # CPR position cache for decoding (icao -> (even_msg, odd_msg, timestamp))
        self._cpr_cache: Dict[str, Tuple[Any, Any, float]] = {}

    def _generate_crc_table(self) -> List[int]:
        """Generate CRC-24 lookup table."""
        table = []
        for i in range(256):
            crc = i << 16
            for _ in range(8):
                if crc & 0x800000:
                    crc = (crc << 1) ^ self.CRC_POLY
                else:
                    crc <<= 1
            table.append(crc & 0xFFFFFF)
        return table

    def _compute_crc(self, data: bytes) -> int:
        """Compute CRC-24 for Mode S message."""
        crc = 0
        for byte in data:
            crc = ((crc << 8) ^ self.CRC_TABLE[(crc >> 16) ^ byte]) & 0xFFFFFF
        return crc

    # -- Demodulation -----------------------------------------------------

    def _plan_slots(self) -> None:
        """How input samples become slot values.

        The decoder reads the magnitude in 0.5 us slots, each input sample
        counting as the mean over its own sample period (a slot's value is
        the area under the input across it). A reply starts anywhere
        between two slot edges, and one that starts half a slot off the
        grid splits every pulse over two slots, so its bits can't be told
        apart. The slots are therefore read at ``_phases`` offsets per slot
        (every 0.125 us, or 0.1 us when 5 divides the samples per slot):
        slot values ``_phases`` times a slot apart are the slots of one
        phase, and each reply is demodulated at the phase that fits it
        best. At exactly 2 MS/s the samples are the slots and the values
        between them would be mixtures of two samples: one phase.

        The offsets repeat every ``_period`` input samples (``_points``
        slot values per period), so each value is a fixed weighting of a
        few consecutive samples: ``_weights[l, k]`` for sample ``_base[l]
        + k`` of the period, for its ``l``-th value.
        """
        ratio = self._ratio
        phases = 1
        if abs(ratio - 1.0) > 1e-9:
            whole = round(ratio)
            phases = 5 if abs(ratio - whole) < 1e-9 and whole % 5 == 0 else 4
        self._phases = phases
        step = Fraction(ratio / phases).limit_denominator(250)
        self._period, self._points = step.numerator, step.denominator
        span = float(step) * phases  # samples per slot
        starts = [float(step * i) for i in range(self._points)]
        self._base = [int(math.floor(x)) for x in starts]
        width = int(math.ceil(span)) + 1
        self._weights = np.zeros((self._points, width), dtype=np.float32)
        for row, (x, base) in enumerate(zip(starts, self._base, strict=True)):
            for k in range(width):
                left, right = max(x, base + k), min(x + span, base + k + 1)
                if right > left:
                    self._weights[row, k] = (right - left) / span
        self._need = max(
            base + int(np.flatnonzero(row)[-1]) + 1
            for base, row in zip(self._base, self._weights, strict=True)
        )
        # The same weights as one matrix: sample j of a period's window ->
        # its values (used when the pattern is small).
        self._matrix = np.zeros((self._need, self._points), dtype=np.float32)
        for row, base in enumerate(self._base):
            for k, weight in enumerate(self._weights[row].tolist()):
                if weight:
                    self._matrix[base + k, row] = weight

    def _to_work_rate(self, mag: np.ndarray) -> np.ndarray:
        """Slot values (mean magnitude over 0.5 us) starting every
        1/``_phases`` of a slot (see :meth:`_plan_slots`).

        Each value is computed from the samples it covers alone, so the
        blocks the input comes in make no difference. Samples that start a
        period not yet complete carry over to the next call.
        """
        if len(self._rest):
            mag = np.concatenate((self._rest, mag))
        period, points, need = self._period, self._points, self._need
        count = (len(mag) - need) // period + 1 if len(mag) >= need else 0
        if count <= 0:
            self._rest = mag.copy()
            return np.zeros(0, dtype=np.float32)
        self._rest = mag[count * period :].copy()
        mag = np.ascontiguousarray(mag, dtype=np.float32)
        if points == 1:
            # Whole samples per offset (8, 10, 20 MS/s...): sums of chunks.
            end = (count + self._phases - 1) * period
            sums = mag[0:end:period].copy()
            for k in range(1, period):
                sums += mag[k:end:period]
            out = sums[:count].copy()
            for q in range(1, self._phases):
                out += sums[q : q + count]
            return out / np.float32(period * self._phases)
        if self._matrix.size <= 8192:
            # Each period's samples (overlapping windows) times the weights.
            windows = np.lib.stride_tricks.as_strided(
                mag,
                shape=(count, need),
                strides=(period * mag.itemsize, mag.itemsize),
                writeable=False,
            )
            return np.einsum("ij,jk->ik", windows, self._matrix).ravel()
        out = np.empty((count, points), dtype=np.float32)
        for row, base in enumerate(self._base):
            col = out[:, row]
            col[:] = 0.0
            for k, weight in enumerate(self._weights[row].tolist()):
                if weight:
                    start = base + k
                    col += (
                        weight * mag[start : start + (count - 1) * period + 1 : period]
                    )
        return out.ravel()

    def _find_preambles(self, m: np.ndarray, count: int) -> np.ndarray:
        """Positions below ``count`` where a Mode S preamble starts in ``m``.

        ``m`` holds slot values (``_phases`` per slot). dump1090's test: the
        pulses in slots 0, 2, 7 and 9 (0, 1, 3.5 and 4.5 us) stand above
        their neighbours, and the quiet slots 4-5 and 11-14 stay below 2/3
        of the mean pulse level.
        """
        p = self._phases
        ok = m[0:count] > m[p : p + count]
        for high, low in (
            (2, 1),
            (2, 3),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (7, 8),
            (9, 8),
            (9, 6),
        ):
            ok &= m[high * p : high * p + count] > m[low * p : low * p + count]
        starts = np.flatnonzero(ok)
        if not len(starts):
            return starts
        high = (
            m[starts] + m[starts + 2 * p] + m[starts + 7 * p] + m[starts + 9 * p]
        ) / 6.0
        for quiet in (4, 5, 11, 12, 13, 14):
            keep = m[starts + quiet * p] < high
            starts, high = starts[keep], high[keep]
        return starts

    def _demodulate(self, m: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """112-bit frames (14 bytes per row) after the preambles at ``starts``.

        Bit n is a 1 when the first half of its 1 us period holds the pulse.
        """
        return np.packbits(self._halves(m, starts, 112)[0], axis=1)

    def _halves(
        self, m: np.ndarray, starts: np.ndarray, bits: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The first ``bits`` bits after the preambles at ``starts`` (True:
        the pulse is in the first half) and how clearly one half stood out
        from the other over the 56 bits every frame has (highest at the
        phase whose slots fit the pulses)."""
        p = self._phases
        first = starts[:, None] + p * (
            self._PREAMBLE_SLOTS + 2 * np.arange(bits)[None, :]
        )
        a, b = m[first], m[first + p]
        clear = min(bits, self.SHORT_MSG_BITS)
        return a > b, np.abs(a[:, :clear] - b[:, :clear]).sum(axis=1)

    # -- Field decoding ---------------------------------------------------

    def _decode_callsign(self, data: bytes) -> str:
        """Decode aircraft identification (callsign)."""
        if len(data) < 7:
            return ""

        # Bytes 1-6 contain callsign (6 bits per character, 8 characters)
        chars = []
        bits = int.from_bytes(data[1:7], "big")

        for i in range(8):
            char_idx = (bits >> (42 - i * 6)) & 0x3F
            if char_idx < len(self.CHARSET):
                char = self.CHARSET[char_idx]
                if char != "#":
                    chars.append(char)

        return "".join(chars).strip()

    def _decode_altitude(self, data: bytes) -> int:
        """Decode altitude from airborne position message."""
        if len(data) < 6:
            return 0

        alt_bits = ((data[1] & 0xFF) << 4) | ((data[2] >> 4) & 0x0F)

        # Q-bit indicates 25 or 100 ft resolution
        q_bit = (alt_bits >> 4) & 0x01

        if q_bit:
            # 25 ft resolution: drop the Q bit (bit 4) and concatenate the 7
            # high bits (11..5) with the 4 low bits (3..0) to form the
            # 11-bit multiplier.
            n = (((alt_bits >> 5) & 0x7F) << 4) | (alt_bits & 0x0F)
            altitude = n * 25 - 1000
        else:
            # Gillham code (100 ft resolution) - simplified
            altitude = alt_bits * 100 - 1300

        return altitude

    @staticmethod
    def _field13(data: bytes) -> int:
        """The 13-bit AC/ID field (message bits 20-32) of a surveillance reply."""
        return ((data[2] & 0x1F) << 8) | data[3]

    def _decode_ac13(self, data: bytes) -> int:
        """Altitude (ft) from the AC13 field of DF0/4/16/20, 0 if unknown.

        Only the common 25 ft coding (M=0, Q=1) is decoded; metric and
        Gillham-coded altitudes read as unknown.
        """
        ac13 = self._field13(data)
        if not ac13 or ac13 & 0x40 or not ac13 & 0x10:
            return 0
        n = ((ac13 & 0x1F80) >> 2) | ((ac13 & 0x20) >> 1) | (ac13 & 0x0F)
        return n * 25 - 1000

    def _decode_squawk(self, data: bytes) -> str:
        """Squawk code from the ID13 field of DF5/21 (e.g. "7000")."""
        id13 = self._field13(data)

        def bit(n: int) -> int:
            return (id13 >> n) & 1

        # Field order, most significant first: C1 A1 C2 A2 C4 A4 X B1 D1 B2
        # D2 B4 D4.
        a = bit(11) | bit(9) << 1 | bit(7) << 2
        b = bit(5) | bit(3) << 1 | bit(1) << 2
        c = bit(12) | bit(10) << 1 | bit(8) << 2
        d = bit(4) | bit(2) << 1 | bit(0) << 2
        return f"{a}{b}{c}{d}"

    def _decode_cpr_position(
        self, icao: str, lat_cpr: int, lon_cpr: int, odd: bool, airborne: bool
    ) -> Tuple[float, float]:
        """
        Decode CPR (Compact Position Reporting) position.

        Requires both odd and even messages for global decode.
        """
        # Check cache for complementary message
        if icao not in self._cpr_cache:
            self._cpr_cache[icao] = (None, None, 0.0)

        even_msg, odd_msg, cache_time = self._cpr_cache[icao]
        current_time = time.time()

        # Cache timeout: 10 seconds
        if current_time - cache_time > 10:
            even_msg = None
            odd_msg = None

        # Store current message
        msg = (lat_cpr, lon_cpr)
        if odd:
            odd_msg = msg
        else:
            even_msg = msg

        self._cpr_cache[icao] = (even_msg, odd_msg, current_time)

        # Need both messages for global decode
        if even_msg is None or odd_msg is None:
            return 0.0, 0.0

        # CPR decoding constants
        if airborne:
            dlat0 = 360.0 / 60
            dlat1 = 360.0 / 59
        else:
            dlat0 = 360.0 / 90
            dlat1 = 360.0 / 89

        lat_even, lon_even = even_msg
        lat_odd, lon_odd = odd_msg

        # Latitude decode
        j = int(np.floor((59 * lat_even - 60 * lat_odd) / 131072.0 + 0.5))
        lat_even_val = dlat0 * ((j % 60) + lat_even / 131072.0)
        lat_odd_val = dlat1 * ((j % 59) + lat_odd / 131072.0)

        if lat_even_val >= 270:
            lat_even_val -= 360
        if lat_odd_val >= 270:
            lat_odd_val -= 360

        # Use most recent message
        if odd:
            lat = lat_odd_val
        else:
            lat = lat_even_val

        # Longitude decode - need NL (longitude zone)
        def nl(lat: float) -> int:
            if abs(lat) >= 87:
                return 1
            return int(
                np.floor(
                    2
                    * np.pi
                    / np.arccos(
                        1 - (1 - np.cos(np.pi / 30)) / (np.cos(np.pi * lat / 180) ** 2)
                    )
                )
            )

        nl_lat = nl(lat)

        if odd:
            ni = max(nl_lat - 1, 1)
            dlon = 360.0 / ni
            m = int(
                np.floor((lon_even * (nl_lat - 1) - lon_odd * nl_lat) / 131072.0 + 0.5)
            )
            lon = dlon * ((m % ni) + lon_odd / 131072.0)
        else:
            ni = max(nl_lat, 1)
            dlon = 360.0 / ni
            m = int(
                np.floor((lon_even * (nl_lat - 1) - lon_odd * nl_lat) / 131072.0 + 0.5)
            )
            lon = dlon * ((m % ni) + lon_even / 131072.0)

        if lon > 180:
            lon -= 360

        return lat, lon

    def _decode_velocity(self, data: bytes) -> Tuple[float, float, int]:
        """Decode velocity from airborne velocity message."""
        if len(data) < 7:
            return 0.0, 0.0, 0

        subtype = data[0] & 0x07

        if subtype in (1, 2):
            # Ground speed
            ew_dir = (data[1] >> 2) & 0x01
            ew_vel = ((data[1] & 0x03) << 8) | data[2]
            ns_dir = (data[3] >> 7) & 0x01
            ns_vel = ((data[3] & 0x7F) << 3) | (data[4] >> 5)

            if ew_vel == 0 and ns_vel == 0:
                return 0.0, 0.0, 0

            ew_vel -= 1
            ns_vel -= 1

            if ew_dir:
                ew_vel = -ew_vel
            if ns_dir:
                ns_vel = -ns_vel

            # Calculate speed and heading
            velocity = np.sqrt(ew_vel**2 + ns_vel**2)
            heading = np.degrees(np.arctan2(ew_vel, ns_vel))
            if heading < 0:
                heading += 360

            # Vertical rate (an all-zero field: no information)
            vr_sign = (data[4] >> 3) & 0x01
            vr_value = ((data[4] & 0x07) << 6) | (data[5] >> 2)
            vertical_rate = (vr_value - 1) * 64 if vr_value else 0
            if vr_sign:
                vertical_rate = -vertical_rate

            return velocity, heading, vertical_rate

        return 0.0, 0.0, 0

    # -- Frame validation -------------------------------------------------

    def _is_known(self, address: int) -> bool:
        """True if ``address`` was seen in a checked reply recently enough."""
        seen = self._known_icao.get(address)
        return seen is not None and self._timestamp - seen <= self.ICAO_TIMEOUT_S

    def _parse_frame(self, frame: bytes) -> Optional[ADSBMessage]:
        """The message in a demodulated frame, or None if it fails its checks."""
        df = frame[0] >> 3
        if df in self._SHORT_FORMATS:
            data = frame[:7]
        elif df in self._LONG_FORMATS:
            data = frame[:14]
        else:
            return None

        residual = self._compute_crc(data[:-3]) ^ int.from_bytes(data[-3:], "big")
        if df in (17, 18):
            if residual:
                return None
            address = int.from_bytes(data[1:4], "big")
            self._known_icao[address] = self._timestamp
        elif df == 11:
            address = int.from_bytes(data[1:4], "big")
            iid = residual & 0x7F
            if residual & ~0x7F or (iid and not self._is_known(address)):
                return None
            if not iid:
                self._known_icao[address] = self._timestamp
        else:
            # Address/parity: the parity is CRC XOR address.
            address = residual
            if not self._is_known(address):
                return None

        icao = f"{address:06X}"
        msg = ADSBMessage(
            protocol=ProtocolType.ADSB,
            timestamp=self._timestamp,
            raw_bits=bytes(data),
            valid=True,
            icao_address=icao,
            downlink_format=df,
        )

        if df in (0, 4, 16, 20):
            msg.altitude = self._decode_ac13(data)
        if df in (5, 21):
            msg.squawk = self._decode_squawk(data)
        if df == 0:
            msg.on_ground = bool(data[0] & 0x04)  # VS: vertical status
        elif df in (4, 5, 20, 21):
            msg.on_ground = (data[0] & 0x07) in (1, 3)  # FS: flight status
        elif df in (11, 17):
            msg.on_ground = (data[0] & 0x07) == 4  # CA: capability

        if df in (17, 18):
            self._parse_extended_squitter(msg, data, icao)
        return msg

    def _parse_extended_squitter(
        self, msg: ADSBMessage, data: bytes, icao: str
    ) -> None:
        """Fill ``msg`` from the ME field of a DF17/18 extended squitter."""
        tc = (data[4] >> 3) & 0x1F
        msg.type_code = tc

        # Aircraft identification (TC 1-4)
        if 1 <= tc <= 4:
            msg.callsign = self._decode_callsign(data[4:])
            msg.category = self.CATEGORIES.get(tc, ("",) * 8)[data[4] & 0x07]

        # Airborne position (TC 9-18)
        elif 9 <= tc <= 18:
            msg.altitude = self._decode_altitude(data[4:])
            msg.on_ground = False

            # CPR position
            lat_cpr = ((data[6] & 0x03) << 15) | (data[7] << 7) | (data[8] >> 1)
            lon_cpr = ((data[8] & 0x01) << 16) | (data[9] << 8) | data[10]
            odd = bool((data[6] >> 2) & 0x01)

            lat, lon = self._decode_cpr_position(icao, lat_cpr, lon_cpr, odd, True)
            if lat != 0.0 or lon != 0.0:
                msg.latitude = lat
                msg.longitude = lon

        # Airborne velocity (TC 19)
        elif tc == 19:
            velocity, heading, vrate = self._decode_velocity(data[4:])
            msg.velocity = velocity
            msg.heading = heading
            msg.vertical_rate = vrate
            msg.vertical_rate_known = bool(((data[8] & 0x07) << 6) | (data[9] >> 2))

        # Surface position (TC 5-8)
        elif 5 <= tc <= 8:
            msg.on_ground = True

    # -- Public API ---------------------------------------------------------

    def decode(self, samples: np.ndarray) -> List[ADSBMessage]:
        """
        Decode ADS-B messages from raw samples.

        Args:
            samples: Magnitude samples from 1090 MHz (complex I/Q is
                converted to magnitude). Consecutive calls are treated as one
                continuous stream.

        Returns:
            List of decoded ADS-B messages (only ones that passed the checks)
        """
        self._messages = []
        samples = np.asarray(samples)
        self._timestamp += len(samples) / self._sample_rate
        if not samples.size:
            return self._messages

        mag = np.abs(samples) if np.iscomplexobj(samples) else samples
        mag = mag.astype(np.float32, copy=False).ravel()
        if not np.all(np.isfinite(mag)):
            mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        # Values no receiver gives (near the float32 limit) can overflow the
        # slot sums: they decode to nothing, without a warning per block.
        with np.errstate(over="ignore", invalid="ignore"):
            self._scan(mag)
        for msg in self._messages:
            self._notify_callbacks(msg)
        return self._messages

    def _scan(self, mag: np.ndarray) -> None:
        """Find and decode the replies in a block of magnitudes (the
        stream's next samples) into ``_messages``."""
        work = self._to_work_rate(mag)
        if len(self._tail):
            work = np.concatenate((self._tail, work))

        # Start positions with room for a whole long frame after them; the
        # rest is scanned next time, when the following samples are there.
        phases = self._phases
        count = len(work) - self._FRAME_SLOTS * phases + 1
        if count <= 0:
            self._tail = work.copy()
            return

        self._forget_old_aircraft()
        starts = self._find_preambles(work, count)
        # A reply passes the preamble test at the phases near its own: the
        # positions closer than a slot to the one before are one reply.
        groups = np.cumsum(np.diff(starts, prepend=-phases) >= phases)
        if len(starts) and starts[-1] >= count - phases:
            # Its other phases may come after ``count``: next time.
            count = int(starts[groups == groups[-1]][0])
            keep = groups != groups[-1]
            starts, groups = starts[keep], groups[keep]
        next_free = self._skip
        for first in range(0, len(starts), self._BATCH):
            batch = starts[first : first + self._BATCH]
            bits, clarity = self._halves(work, batch, 112)
            frames = np.packbits(bits, axis=1)
            # The clearest phase of each reply; formats that are never
            # accepted are not parsed.
            best = self._clearest(groups[first : first + self._BATCH], clarity)
            df = frames[best, 0] >> 3
            for index in best[np.isin(df, self._FORMATS)].tolist():
                start = int(batch[index])
                if start < next_free:
                    continue  # inside a frame already decoded
                msg = self._parse_frame(frames[index].tobytes())
                if msg is None:
                    continue
                self._messages.append(msg)
                next_free = start + phases * (
                    self._PREAMBLE_SLOTS + 16 * len(msg.raw_bits)
                )

        self._tail = work[count:].copy()
        self._skip = max(0, next_free - count)

    _FORMATS = np.array(_SHORT_FORMATS + _LONG_FORMATS)

    @staticmethod
    def _clearest(groups: np.ndarray, clarity: np.ndarray) -> np.ndarray:
        """Index of the clearest candidate of each group (consecutive runs
        of equal ``groups``), in order."""
        if not len(groups):
            return np.zeros(0, dtype=np.int64)
        order = np.lexsort((-clarity, groups))
        first = np.flatnonzero(np.diff(groups[order], prepend=groups[order][0] - 1))
        return np.sort(order[first])

    def _forget_old_aircraft(self) -> None:
        """Drop addresses not seen for longer than ``ICAO_TIMEOUT_S``."""
        cutoff = self._timestamp - self.ICAO_TIMEOUT_S
        stale = [a for a, seen in self._known_icao.items() if seen < cutoff]
        for address in stale:
            del self._known_icao[address]

    def reset(self) -> None:
        """Reset decoder state."""
        self._cpr_cache.clear()
        self._known_icao.clear()
        self._timestamp = 0.0
        self._rest = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(0, dtype=np.float32)
        self._skip = 0


# ============================================================================
# FLEX Protocol Decoder
# ============================================================================


@dataclass
class FLEXMessage(DecodedMessage):
    """FLEX pager message."""

    capcode: int = 0  # Pager address
    cycle: int = 0  # FLEX cycle number (0-14)
    frame: int = 0  # Frame within cycle (0-127)
    phase: str = ""  # Phase (A, B, C, D)
    message_type: str = ""  # "alpha", "numeric", "tone", "secure"
    content: str = ""
    baud_rate: int = 1600


class FLEXDecoder(ProtocolDecoder):
    """
    FLEX pager protocol decoder.

    FLEX is a high-speed paging protocol developed by Motorola. Every
    1.875 s frame starts with a 1600 bps header: bit sync, a 64-bit sync
    pattern naming the data speed (a 16-bit code, the marker 0xA6C6AAAA
    and the code inverted) and the frame information word (FIW: cycle and
    frame number). 11 blocks of data follow, each 8 codewords interleaved
    bit by bit; at 1600 bps 2-level FSK they form one phase of 88
    codewords: the block information word (BIW), address words, vector
    words (page type and where its message words are) and message words.
    Codewords are BCH(31,21) with a parity bit, sent least significant
    bit first.

    Input is an FM-discriminator baseband (either polarity) at any sample
    rate; consecutive ``decode()`` calls are one continuous stream.

    Features:
    - Sync detection with bit errors and either FSK polarity
    - Clock recovery and integrate-and-dump bit decisions (see BitSlicer)
    - BCH(31,21) correction of up to two bit errors per codeword
    - Short and long addresses (capcodes)
    - Alphanumeric, numeric and tone-only pages
    - 1600 bps 2-level frames (the other speeds are recognised and counted
      in ``unsupported_frames``, not decoded)
    """

    # FLEX constants: the marker in the middle of the 64-bit sync pattern
    # (SYNC_1) and its complement (SYNC_2, as sent with inverted polarity).
    SYNC_1 = 0xA6C6AAAA
    SYNC_2 = 0x5939AAAA
    MARKER = 0xA6C6AAAA
    # Sync code (first 16 bits of the pattern) -> (bits/s, FSK levels), per
    # ARIB STD-43A (as multimon-ng reads them): A1, A2, A3 (1600 baud),
    # A4 (3200 baud) and A7 (ReFLEX, as A4).
    SYNC_CODES: Dict[int, Tuple[int, int]] = {
        0x870C: (1600, 2),
        0x7B18: (3200, 2),
        0xB068: (3200, 4),
        0xDEA0: (6400, 4),
        0x4C7C: (6400, 4),
    }

    # Frame structure
    FRAME_SYNC_BITS = 64
    FRAME_INFO_BITS = 32
    BLOCK_BITS = 256  # 8 interleaved codewords per block (1600 bps)
    BLOCKS_PER_FRAME = 11  # Data blocks per frame
    WORDS_PER_PHASE = 88
    # Bits skipped after the sync pattern before the FIW, and the second
    # sync (25 ms) before the data, at 1600 bps.
    _FIW_GAP_BITS = 16
    _SYNC2_BITS = 40

    # BCH(31,21) polynomial
    BCH_POLY = 0x769

    # Page types in a vector word.
    PAGE_TYPES = (
        "secure",
        "instruction",
        "tone",
        "numeric",
        "numeric",
        "alpha",
        "binary",
        "numeric",
    )
    _NUMERIC_CHARS = "0123456789 U -]["

    _SYNC1, _FIW, _SYNC2, _DATA = range(4)

    def __init__(
        self, sample_rate: float, baud_rate: int = 1600, matched_filter: bool = True
    ):
        """
        Initialize FLEX decoder.

        Args:
            sample_rate: Sample rate in Hz
            baud_rate: Data speed of the frames expected (1600, 3200 or
                6400 bits/s); the speed each frame names is used
            matched_filter: Average each bit over its period before deciding
                it (leave on for a discriminator output)
        """
        super().__init__(sample_rate)
        self._baud_rate = baud_rate
        self._samples_per_bit = int(sample_rate / baud_rate)
        if self._samples_per_bit < 1:
            raise ValueError(
                f"sample_rate {sample_rate:g} Hz is too low for FLEX at "
                f"{baud_rate} baud; need at least {baud_rate} Hz"
            )
        # The frame header is always 1600 bps 2-level FSK.
        self._slicer = BitSlicer(sample_rate, 1600, matched_filter)

        # State
        self._state = self._SYNC1
        self._reg = 0
        self._count = 0
        self._bits: List[int] = []
        self._data_invert = 0  # 1: data bits are the complement of the slicer's
        self._phase = "A"
        self._cycle = 0
        self._frame_num = 0

        # Messages
        self._messages: List[FLEXMessage] = []
        self._timestamp = 0.0
        self.frames_seen = 0
        self.frames_failed = 0
        self.unsupported_frames = 0
        self.last_unsupported = ""

    @property
    def baud_rate(self) -> int:
        """Get baud rate."""
        return self._baud_rate

    def _bits_to_word(self, bits: Sequence[int]) -> int:
        """Convert 32 bits to word."""
        word = 0
        for bit in bits[:32]:
            word = (word << 1) | bit
        return word

    @staticmethod
    def _word_value(bits: Sequence[int]) -> Tuple[bool, int]:
        """``(ok, value)`` of one codeword from its 32 bits in the order
        sent: the 21 data bits (the first one sent is bit 0 of ``value``)
        after BCH correction.

        The even parity bit counts as one more error when it disagrees, and
        more than two errors reject the word: three errors would otherwise
        often be "corrected" into a different codeword.
        """
        codeword = 0
        for bit in bits[:31]:
            codeword = (codeword << 1) | bit
        ok, fixed, errors = bch3121_correct(codeword)
        if ok and len(bits) > 31:
            errors += (bin(fixed).count("1") & 1) != bits[31]
            ok = errors <= 2
        data = fixed >> 10
        value = 0
        for _ in range(21):
            value = (value << 1) | (data & 1)
            data >>= 1
        return ok, value

    def _check_bch(self, word: int) -> Tuple[bool, int]:
        """Check BCH(31,21) error correction of a 32-bit word whose bits
        31..1 are the codeword in the order sent (bit 31 first) and bit 0
        is the parity bit; up to two wrong bits are corrected."""
        ok, fixed, _errors = bch3121_correct(word >> 1)
        if not ok:
            return False, word
        return True, (fixed << 1) | (word & 1)

    @staticmethod
    def _checksum_ok(value: int) -> bool:
        """The 4-bit checksum of a frame or block information word: the
        nibbles of bits 0-19 plus bit 20 add up to 0xF."""
        total = value >> 20 & 1
        for shift in range(0, 20, 4):
            total += value >> shift & 0xF
        return total & 0xF == 0xF

    def _decode_alpha(self, words: Sequence[int], skip_first: bool = False) -> str:
        """Characters of an alphanumeric page: three 7-bit characters per
        message word (bits 0-6, 7-13 and 14-20); ETX (0x03) is filler. With
        ``skip_first`` the first character (the signature of a page's first
        fragment) is not text."""
        chars = []
        for index, word in enumerate(words):
            for slot in range(3):
                if skip_first and index == 0 and slot == 0:
                    continue
                code = (word >> (7 * slot)) & 0x7F
                if code == 0x03:
                    continue
                if 32 <= code < 127:
                    chars.append(chr(code))
                elif code in (0x0A, 0x0D):
                    chars.append(" ")
        return "".join(chars).strip()

    def _decode_numeric(self, words: Sequence[int], skip_bits: int = 2) -> str:
        """Digits of a numeric page: 4-bit characters (least significant bit
        first) packed across the 21 bits of each word, after ``skip_bits``
        header bits; 0xC is filler."""
        chars = []
        digit = 0
        count = 4 + skip_bits
        for word in words:
            for _ in range(21):
                digit = (digit >> 1) & 0x07
                if word & 1:
                    digit |= 0x08
                word >>= 1
                count -= 1
                if count == 0:
                    if digit != 0x0C:
                        chars.append(self._NUMERIC_CHARS[digit])
                    count = 4
        return "".join(chars).strip()

    @staticmethod
    def _capcode(first: int, second: Optional[int]) -> Tuple[int, bool]:
        """``(capcode, long address)`` from an address word (and the next
        one, for a long address)."""
        long_address = (
            first < 0x008001 or 0x1E0000 < first < 0x1F0001 or first > 0x1F7FFE
        )
        if not long_address or second is None:
            return first - 0x8000, long_address
        return first + ((second ^ 0x1FFFFF) << 15) + 0x1F9000, True

    def _parse_frame(self, frame_bits: Sequence[int]) -> List[FLEXMessage]:
        """Pages in the data of one 1600 bps frame (2816 bits as sent)."""
        messages: List[FLEXMessage] = []
        if len(frame_bits) < self.BLOCKS_PER_FRAME * self.BLOCK_BITS:
            return messages
        bits = np.asarray(frame_bits[: self.BLOCKS_PER_FRAME * self.BLOCK_BITS])
        # Block b holds 8 codewords, sent one bit of each in turn.
        by_word = bits.reshape(self.BLOCKS_PER_FRAME, 32, 8).transpose(0, 2, 1)
        by_word = by_word.reshape(self.WORDS_PER_PHASE, 32).tolist()
        values: List[int] = []
        good: List[bool] = []
        for word_bits in by_word:
            ok, value = self._word_value(word_bits)
            values.append(value)
            good.append(ok)

        biw = values[0]
        if not good[0]:
            self.frames_failed += 1
            return messages
        if biw in (0, 0x1FFFFF):
            return messages  # an idle frame
        first_address = ((biw >> 8) & 0x3) + 1
        first_vector = (biw >> 10) & 0x3F
        if first_vector <= first_address - 1 or first_vector >= self.WORDS_PER_PHASE:
            self.frames_failed += 1
            return messages

        i = first_address
        while i < first_vector:
            address = values[i]
            if address in (0, 0x1FFFFF):
                i += 1
                continue
            second = values[i + 1] if i + 1 < first_vector else None
            capcode, long_address = self._capcode(address, second)
            if long_address and second is None:
                break  # a long address needs two words; the field ended
            used = [i] + ([i + 1] if long_address else [])
            vector_index = first_vector + (i - first_address)
            i += 2 if long_address else 1
            if vector_index >= self.WORDS_PER_PHASE:
                break
            vector = values[vector_index]
            used.append(vector_index)
            page_type = self.PAGE_TYPES[(vector >> 4) & 0x7]
            start = (vector >> 7) & 0x7F
            length = (vector >> 14) & 0x7F
            content = ""
            if page_type == "alpha":
                header = start
                if long_address:
                    header = vector_index + 1
                    used.append(header)
                    words = list(range(start, min(start + length - 1, 88)))
                else:
                    words = list(range(start + 1, min(start + length, 88)))
                    used.append(header)
                if header >= self.WORDS_PER_PHASE:
                    continue
                fragment = (values[header] >> 11) & 0x3
                used += words
                content = self._decode_alpha(
                    [values[w] for w in words], skip_first=fragment == 3
                )
            elif page_type == "numeric":
                count = ((vector >> 14) & 0x7) + 1  # message words
                if long_address:
                    # The first message word is the second vector word.
                    words = [vector_index + 1] + list(range(start, start + count - 1))
                else:
                    words = list(range(start, start + count))
                words = [w for w in words if w < self.WORDS_PER_PHASE]
                used += words
                skip = 10 if (vector >> 4) & 0x7 == 7 else 2
                content = self._decode_numeric([values[w] for w in words], skip)
            elif page_type == "instruction":
                continue
            bad = sum(1 for w in used if w < len(good) and not good[w])
            error = ""
            if bad:
                error = f"{bad} codeword{'s' if bad != 1 else ''} failed the BCH check"
                self.frames_failed += 1
            msg = FLEXMessage(
                protocol=ProtocolType.FLEX,
                timestamp=self._timestamp,
                raw_bits=b"".join(int(values[w]).to_bytes(3, "big") for w in used),
                valid=not bad,
                error_message=error,
                capcode=capcode,
                cycle=self._cycle,
                frame=self._frame_num,
                phase=self._phase,
                message_type=page_type,
                content=content,
                baud_rate=1600,
            )
            messages.append(msg)
        return messages

    def _sync_mode(self, reg: int) -> Optional[Tuple[int, bool]]:
        """``(code, inverted)`` if the last 64 bits are a sync pattern."""
        for inverted in (False, True):
            value = reg ^ 0xFFFFFFFFFFFFFFFF if inverted else reg
            marker = (value >> 16) & 0xFFFFFFFF
            if (marker ^ self.MARKER).bit_count() >= 4:
                continue
            high = value >> 48
            low = value & 0xFFFF
            if (high ^ low ^ 0xFFFF).bit_count() >= 4:
                continue
            return high, inverted
        return None

    def _read_fiw(self, bits: Sequence[int]) -> bool:
        """Check the frame information word (32 bits as sent) and keep its
        cycle and frame numbers. The data polarity is confirmed here: the
        checksum fails for complemented bits."""
        for invert in (self._data_invert, 1 - self._data_invert):
            ok, value = self._word_value([b ^ invert for b in bits])
            if ok and self._checksum_ok(value):
                self._data_invert = invert
                self._cycle = (value >> 4) & 0xF
                self._frame_num = (value >> 8) & 0x7F
                return True
        return False

    def decode(self, samples: np.ndarray) -> List[FLEXMessage]:
        """
        Decode FLEX messages from FSK-demodulated samples.

        Args:
            samples: FSK-demodulated samples

        Returns:
            List of decoded FLEX messages
        """
        x = _as_real(samples)
        self._messages = []
        self._timestamp += len(x) / self._sample_rate
        for bit in self._slicer.process(x):
            self._feed_bit(bit)
        return self._messages

    def _feed_bit(self, bit: int) -> None:
        state = self._state
        if state == self._SYNC1:
            self._reg = ((self._reg << 1) | bit) & 0xFFFFFFFFFFFFFFFF
            if ((self._reg >> 16 ^ self.MARKER) & 0xFFFFFFFF).bit_count() >= 4 and (
                (self._reg >> 16 ^ ~self.MARKER) & 0xFFFFFFFF
            ).bit_count() >= 4:
                return  # (quick reject before the full check)
            found = self._sync_mode(self._reg)
            if found is None:
                return
            code, inverted = found
            mode = None
            for known, value in self.SYNC_CODES.items():
                if (known ^ code).bit_count() <= 2:
                    mode = value
                    break
            if mode is None:
                return
            self.frames_seen += 1
            if mode != (1600, 2):
                self.unsupported_frames += 1
                self.last_unsupported = f"{mode[0]} bps {mode[1]}-level"
                self._reg = 0
                return
            # The sync pattern is sent with the opposite polarity to the
            # data (as multimon-ng reads FLEX).
            self._data_invert = 0 if inverted else 1
            self._state, self._count, self._bits = self._FIW, 0, []
            self._reg = 0
        elif state == self._FIW:
            self._count += 1
            if self._count <= self._FIW_GAP_BITS:
                return
            self._bits.append(bit)
            if len(self._bits) < 32:
                return
            if self._read_fiw(self._bits):
                self._state, self._count = self._SYNC2, 0
            else:
                self.frames_failed += 1
                self._state = self._SYNC1
            self._bits = []
        elif state == self._SYNC2:
            self._count += 1
            if self._count >= self._SYNC2_BITS:
                self._state, self._bits = self._DATA, []
        else:
            self._bits.append(bit ^ self._data_invert)
            if len(self._bits) < self.BLOCKS_PER_FRAME * self.BLOCK_BITS:
                return
            self._baud_rate = 1600
            for msg in self._parse_frame(self._bits):
                self._messages.append(msg)
                self._notify_callbacks(msg)
            self._bits = []
            self._state = self._SYNC1

    def reset(self) -> None:
        """Reset decoder state."""
        self._slicer.reset()
        self._state = self._SYNC1
        self._reg = 0
        self._count = 0
        self._bits = []
        self._messages = []
        self._timestamp = 0.0
        self.unsupported_frames = 0
        self.last_unsupported = ""


# ============================================================================
# ACARS (Aircraft Communications Addressing and Reporting System) Decoder
# ============================================================================


@dataclass
class ACARSMessage(DecodedMessage):
    """ACARS decoded message."""

    mode: str = ""  # Mode character
    registration: str = ""  # Aircraft registration (7 chars)
    ack: str = ""  # Acknowledgement character
    label: str = ""  # Message label (2 chars)
    block_id: str = ""  # Block identifier
    message_number: str = ""  # Message sequence number
    flight_id: str = ""  # Flight identification
    text: str = ""  # Message text
    block_end: str = ""  # Block end character


class ACARSDecoder(ProtocolDecoder):
    """
    ACARS decoder.

    ACARS is a digital data link system for aircraft communications
    on VHF frequencies (129.125, 130.025, 130.450 MHz etc.).

    The 2400 bit/s data is MSK on an AM carrier: the audio is a 1200 Hz or
    a 2400 Hz tone for each bit period, 2400 Hz when the bit equals the
    previous one and 1200 Hz when it changes (as acarsdec demodulates it).
    Characters are 7-bit ASCII with odd parity, least significant bit
    first: pre-key, '+' '*' (bit sync), SYN SYN, SOH, mode, address,
    acknowledgement, label, block ID, STX and the text, ETX or ETB, a
    CRC-16 and DEL.

    Input, at any sample rate, is either the data itself from a coherent
    MSK demodulator (the default: one sign per bit value, the polarity is
    found from the SYN characters), or with ``differential`` a tone
    baseband (positive while 2400 Hz is heard, negative for 1200 Hz), from
    which the bits are rebuilt; there one wrong tone inverts the rest of
    the block, so a coherent demodulator is better. Consecutive
    ``decode()`` calls are one continuous stream.

    Features:
    - Clock recovery and integrate-and-dump bit decisions (see BitSlicer)
    - Bit-level SYN SYN search in either polarity
    - Parity check and CRC-16 validation
    - Flight ID and registration extraction
    - Message text decoding
    """

    # ACARS constants (7-bit values; the 8th bit sent is odd parity)
    PREKEY = 0x2B2B  # Preamble: ++
    SYN = 0x16
    SOH = 0x01  # Start of header
    STX = 0x02  # Start of text
    ETX = 0x03  # End of text
    ETB = 0x17  # End of transmission block
    DEL = 0x7F  # Block check sequence suffix
    NAK = 0x15

    # Character set (odd parity, 7-bit ASCII)
    BAUD_RATE = 2400
    MAX_BLOCK = 240  # characters from mode to ETX/ETB

    _SEARCH, _SOH, _TEXT, _CRC = range(4)

    def __init__(
        self,
        sample_rate: float,
        matched_filter: bool = True,
        differential: bool = False,
    ):
        """
        Initialize ACARS decoder.

        Args:
            sample_rate: Sample rate in Hz
            matched_filter: Average each bit over its period before deciding
                it (leave on for a demodulator output)
            differential: The input is a tone baseband (2400 Hz positive)
                rather than the data bits
        """
        super().__init__(sample_rate)
        self._samples_per_bit = int(sample_rate / self.BAUD_RATE)
        if self._samples_per_bit < 1:
            raise ValueError(
                f"sample_rate {sample_rate:g} Hz is too low for ACARS at "
                f"{self.BAUD_RATE} baud; need at least {self.BAUD_RATE} Hz"
            )
        self._slicer = BitSlicer(sample_rate, self.BAUD_RATE, matched_filter)
        self._differential = differential

        # State
        self._last_bit = 0  # last data bit (differential decoding)
        self._state = self._SEARCH
        self._reg = 0  # last 16 data bits, the newest in bit 15
        self._invert = 0
        self._byte = 0
        self._nbits = 0
        self._block = bytearray()
        self._crc_bytes = bytearray()

        # Messages
        self._messages: List[ACARSMessage] = []
        self._timestamp = 0.0
        self.frames_seen = 0
        self.frames_failed = 0

    def _remove_parity(self, byte: int) -> int:
        """Remove parity bit and return 7-bit character."""
        return byte & 0x7F

    @staticmethod
    def with_parity(char: int) -> int:
        """``char`` (7 bits) with its odd parity bit (bit 7)."""
        char &= 0x7F
        return char | (0x80 if bin(char).count("1") % 2 == 0 else 0)

    def _compute_crc(self, data: bytes) -> int:
        """ACARS block check: CRC-16 (CCITT polynomial, reflected, initial
        value 0) over the characters as sent, parity bits included.
        Running it over a block and its two CRC bytes gives 0."""
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
        return crc

    def _parse_message(
        self, data: bytes, crc_ok: bool = True, parity_errors: int = 0
    ) -> Optional[ACARSMessage]:
        """Parse an ACARS block from its characters (as sent).

        ``data`` starts at the mode character (anything up to SOH before
        it is skipped) and runs to ETX/ETB.
        """
        soh = data.find(bytes([self.SOH]))
        if 0 <= soh < 3:
            data = data[soh + 1 :]
        if len(data) < 12:
            return None
        chars = [b & 0x7F for b in data]

        def text(part: Sequence[int]) -> str:
            return "".join(chr(c) for c in part if 32 <= c < 127)

        mode = text(chars[0:1])
        registration = text(chars[1:8]).lstrip(".").strip()
        ack = "!" if chars[8] == self.NAK else text(chars[8:9])
        label = text(chars[9:11])
        block_id = text(chars[11:12])
        msg_number = ""
        flight_id = ""
        body = ""
        block_end = ""
        end = len(chars)
        for i in range(12, len(chars)):
            if chars[i] in (self.ETX, self.ETB):
                end = i
                block_end = chr(chars[i])
                break
        if len(chars) > 12 and chars[12] == self.STX:
            body = text(chars[13:end])
            # A downlink (block ID a digit) starts with the message number
            # and the flight ID.
            if block_id.isdigit() and len(body) >= 10:
                msg_number, flight_id, body = body[:4], body[4:10], body[10:]
        problems = []
        if not crc_ok:
            problems.append("CRC mismatch")
        if parity_errors:
            problems.append(
                f"{parity_errors} parity error{'s' if parity_errors != 1 else ''}"
            )
        return ACARSMessage(
            protocol=ProtocolType.ACARS,
            timestamp=self._timestamp,
            raw_bits=bytes(data),
            valid=crc_ok,
            error_message=", ".join(problems),
            mode=mode,
            registration=registration,
            ack=ack,
            label=label,
            block_id=block_id,
            message_number=msg_number.strip(),
            flight_id=flight_id.strip(),
            text=body.strip(),
            block_end=block_end,
        )

    def decode(self, samples: np.ndarray) -> List[ACARSMessage]:
        """
        Decode ACARS messages from MSK-demodulated samples.

        Args:
            samples: MSK-demodulated samples (the data bits; with
                ``differential``, positive for 2400 Hz)

        Returns:
            List of decoded ACARS messages (blocks that fail the CRC are
            returned with ``valid`` False)
        """
        x = _as_real(samples)
        self._messages = []
        self._timestamp += len(x) / self._sample_rate
        for value in self._slicer.process(x):
            if self._differential:
                # 2400 Hz: the same bit again; 1200 Hz: the bit changes.
                value = self._last_bit if value else self._last_bit ^ 1
                self._last_bit = value
            self._feed_bit(value)
        return self._messages

    def _feed_bit(self, bit: int) -> None:
        state = self._state
        if state == self._SEARCH:
            self._reg = (self._reg >> 1) | (bit << 15)
            if self._reg == 0x1616:
                self._invert = 0
            elif self._reg == 0xE9E9:
                self._invert = 1
            else:
                return
            self._state, self._byte, self._nbits = self._SOH, 0, 0
            return
        self._byte = (self._byte >> 1) | ((bit ^ self._invert) << 7)
        self._nbits += 1
        if self._nbits < 8:
            return
        byte, self._byte, self._nbits = self._byte, 0, 0
        if state == self._SOH:
            if byte == self.SOH:
                self._state = self._TEXT
                self._block = bytearray()
            elif byte != self.SYN:
                self._restart()
        elif state == self._TEXT:
            self._block.append(byte)
            if byte & 0x7F in (self.ETX, self.ETB) and len(self._block) >= 13:
                self._state = self._CRC
                self._crc_bytes = bytearray()
            elif len(self._block) > self.MAX_BLOCK:
                self._restart()
        else:
            self._crc_bytes.append(byte)
            if len(self._crc_bytes) == 2:
                self._end_block()

    def _restart(self) -> None:
        self._state = self._SEARCH
        self._reg = 0
        self._block = bytearray()

    def _end_block(self) -> None:
        block = bytes(self._block)
        crc_ok = self._compute_crc(block + bytes(self._crc_bytes)) == 0
        parity_errors = sum(1 for b in block if bin(b).count("1") % 2 == 0)
        self._restart()
        if not crc_ok and parity_errors > max(2, len(block) // 4):
            return  # noise that happened to look like SYN SYN SOH
        msg = self._parse_message(block, crc_ok, parity_errors)
        self.frames_seen += 1
        if msg is None:
            self.frames_failed += 1
            return
        if not msg.valid:
            self.frames_failed += 1
        self._messages.append(msg)
        self._notify_callbacks(msg)

    def reset(self) -> None:
        """Reset decoder state."""
        self._slicer.reset()
        self._last_bit = 0
        self._restart()
        self._messages = []
        self._timestamp = 0.0


# Factory function
def create_protocol_decoder(
    protocol: ProtocolType, sample_rate: float, **kwargs
) -> ProtocolDecoder:
    """
    Create a protocol decoder.

    Args:
        protocol: Protocol type
        sample_rate: Sample rate in Hz
        **kwargs: Protocol-specific parameters

    Returns:
        Protocol decoder instance

    Supported protocols:
        - POCSAG: Paging (512/1200/2400 baud FSK)
        - FLEX: Paging (1600/3200/6400 baud FSK)
        - AX25/APRS: Amateur packet radio (1200 baud AFSK)
        - RDS: FM broadcast data (1187.5 baud BPSK)
        - ADSB: Aircraft transponder (1090 MHz PPM)
        - ACARS: Aircraft communications (2400 baud MSK)
    """
    if protocol == ProtocolType.POCSAG:
        return POCSAGDecoder(sample_rate, **kwargs)
    elif protocol == ProtocolType.FLEX:
        return FLEXDecoder(sample_rate, **kwargs)
    elif protocol in (ProtocolType.AX25, ProtocolType.APRS):
        return AX25Decoder(sample_rate, **kwargs)
    elif protocol == ProtocolType.RDS:
        return RDSDecoder(sample_rate)
    elif protocol == ProtocolType.ADSB:
        return ADSBDecoder(sample_rate)
    elif protocol == ProtocolType.ACARS:
        return ACARSDecoder(sample_rate)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


def demodulate_for_protocol(
    samples: np.ndarray, protocol: ProtocolType
) -> Optional[np.ndarray]:
    """Demodulate I/Q into a baseband for a protocol decoder (low level).

    The decoders operate on demodulated baseband, not raw I/Q. This returns
    the pulse envelope for ADS-B and an FM discriminator output for the
    others, which is what the POCSAG and FLEX decoders (FSK) and direct-FSK
    packet want. It does not select a channel, so give it I/Q that holds
    just the channel. AX.25 at 1200 baud (APRS) is AFSK and ACARS is MSK on
    AM: their decoders need a tone detector's or an MSK demodulator's
    output, which a discriminator is not. For raw receiver I/Q of any
    protocol use :func:`sdr_module.dsp.live_decoders.create_live_decoder`,
    which selects the channel and demodulates each protocol properly.

    Returns None when the required baseband cannot be produced for the protocol
    (currently RDS, which needs 57 kHz subcarrier recovery), so callers can say
    so rather than silently decoding nothing.
    """
    samples = np.asarray(samples)

    if protocol == ProtocolType.ADSB:
        return np.abs(samples).astype(np.float32)

    fsk_like = {
        ProtocolType.POCSAG,
        ProtocolType.FLEX,
        ProtocolType.AX25,
        ProtocolType.APRS,
        ProtocolType.ACARS,
    }
    if protocol in fsk_like:
        if np.iscomplexobj(samples) and np.any(samples.imag != 0):
            # FM discriminator: instantaneous frequency of the I/Q stream.
            return np.angle(samples[1:] * np.conj(samples[:-1])).astype(np.float32)
        # Already a real, pre-demodulated baseband.
        return samples.real.astype(np.float32)

    # RDS needs 57 kHz subcarrier recovery + coherent BPSK, not implemented here.
    return None


# Export all decoders and message types
__all__ = [
    # Protocol types
    "ProtocolType",
    "DecodedMessage",
    # POCSAG
    "POCSAGMessage",
    "POCSAGDecoder",
    # FLEX
    "FLEXMessage",
    "FLEXDecoder",
    # AX.25/APRS
    "AX25Frame",
    "APRSMessage",
    "AX25Decoder",
    "parse_aprs_info",
    "APRS_TYPE_NAMES",
    "APRS_SYMBOLS",
    # RDS
    "RDSData",
    "RDSDecoder",
    "rds_char",
    # ADS-B
    "ADSBMessage",
    "ADSBDecoder",
    # ACARS
    "ACARSMessage",
    "ACARSDecoder",
    # Factory
    "create_protocol_decoder",
    "demodulate_for_protocol",
    # Building blocks
    "BitSlicer",
    "bch3121_encode",
    "bch3121_syndrome",
    "bch3121_correct",
]

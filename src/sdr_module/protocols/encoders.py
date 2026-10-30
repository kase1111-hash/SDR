"""
Specific protocol encoder implementations.

Ready-to-use encoders for RTTY, Morse code, ASCII FSK and PSK31. Each turns
text into complex I/Q at ``config.sample_rate`` with the signal centred on
``config.carrier_freq`` (an audio offset: 1000 Hz puts it 1 kHz above 0 Hz).

They are built on :mod:`sdr_module.protocols.text_modes`, the same code
that Transmit Text uses, so the library, the command line and the GUI send
exactly the same signals:

* RTTY: ITA2 with automatic LTRS/FIGS shifts (US-TTY figures), 45.45 Bd and
  170 Hz shift unless the config says otherwise, 1.5 stop bits, continuous
  phase, mark above the carrier;
* Morse: the full ITU character set and prosigns (``<SK>``), PARIS timing;
* ASCII: 8N1 FSK of the ASCII bytes;
* PSK31: the full case-sensitive G3PLX Varicode with cosine-shaped phase
  reversals.

Characters a protocol can't send are left out and reported:
:attr:`skipped` after :meth:`encode_text` (character -> times it was
left out), :meth:`unsupported_chars` before.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .encoder import EncoderConfig, ModulationType, ProtocolEncoder
from .text_modes import (
    ASCII_BAUD,
    BAUD_RANGE,
    ITA2_FIGURES_US,
    ITA2_LETTERS,
    MODE_ASCII,
    MODE_CW,
    MODE_PSK31,
    MODE_RTTY,
    MORSE_CODES,
    PSK31_BAUD,
    RTTY_BAUD,
    RTTY_SHIFT_HZ,
    SHIFT_RANGE,
    VARICODE,
    WPM_RANGE,
    TextCheck,
    TextSettings,
    TextSignal,
    check_text,
    format_number,
)


def _check_range(value: float, bounds: tuple, what: str, unit: str) -> None:
    """A value the encoders can't send is refused, never quietly changed."""
    low, high = bounds
    if not low <= float(value) <= high:
        raise ValueError(
            f"{what} must be {format_number(low)} to {format_number(high)} {unit}, "
            f"not {format_number(value)}"
        )


class _TextEncoder(ProtocolEncoder):
    """Shared by the text encoders: builds the :class:`TextSignal` and moves
    it to the carrier; remembers what was left out."""

    MODE = MODE_RTTY

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        #: Characters left out by the last :meth:`encode_text` (char -> count).
        self.skipped: Dict[str, int] = {}
        #: What the last :meth:`encode_text` sent (:class:`TextCheck`).
        self.last_check: Optional[TextCheck] = None

    def _settings(self) -> TextSettings:  # pragma: no cover - overridden
        raise NotImplementedError

    def unsupported_chars(self, text: str) -> Dict[str, int]:
        """The characters of ``text`` this protocol can't send (char ->
        how often it occurs)."""
        return dict(check_text(text, self.MODE).unsupported)

    def text_signal(self, text: str) -> TextSignal:
        """``text`` as a :class:`TextSignal` with this encoder's settings."""
        return TextSignal(text, self._settings())

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to complex I/Q centred on the carrier.

        Args:
            text: Text to encode

        Returns:
            Complex I/Q samples (complex64); empty when there is nothing
            this protocol can send. :attr:`skipped` lists what was left out.
        """
        signal = self.text_signal(text)
        self.skipped = dict(signal.check.unsupported)
        self.last_check = signal.check
        if signal.empty:
            return np.zeros(0, dtype=np.complex64)
        t = signal.times(self._sample_rate)
        out = signal.analytic(t, self._carrier_freq) * self._config.amplitude
        return out.astype(np.complex64)

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Encode bytes (decoded as ASCII text first; empty if they aren't)."""
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            return np.array([], dtype=np.complex64)
        return self.encode_text(text)


class RTTYEncoder(_TextEncoder):
    """
    Radio Teletype (RTTY) encoder.

    FSK with ITA2 (Baudot) 5-bit codes: a start bit, five data bits and 1.5
    stop bits per character. The standard amateur setting, 45.45 Bd with a
    170 Hz shift, is used unless the config sets ``baud_rate`` and
    ``frequency_shift``. Letters and figures shifts are inserted
    automatically.
    """

    MODE = MODE_RTTY

    #: ITA2 letters case: character -> code (first-sent bit is bit 0).
    BAUDOT_LETTERS: Dict[str, int] = {
        c: i
        for i, c in enumerate(ITA2_LETTERS)
        if c is not None and (c.isalpha() or c == " ")
    }
    #: ITA2 figures case (US TTY): character -> code.
    BAUDOT_FIGURES: Dict[str, int] = {
        c: i
        for i, c in enumerate(ITA2_FIGURES_US)
        if c is not None and c not in "\n\r "
    }
    LTRS = 0b11111
    FIGS = 0b11011

    def __init__(self, config: EncoderConfig):
        """Initialize RTTY encoder with 45.45 baud default."""
        if config.baud_rate == 0:
            config.baud_rate = RTTY_BAUD  # Standard RTTY baud rate
        if not config.frequency_shift:
            config.frequency_shift = RTTY_SHIFT_HZ  # Standard shift
        _check_range(config.baud_rate, BAUD_RANGE, "The RTTY baud rate", "Bd")
        _check_range(config.frequency_shift, SHIFT_RANGE, "The RTTY shift", "Hz")
        super().__init__(config)

    def _settings(self) -> TextSettings:
        assert self._config.frequency_shift is not None
        return TextSettings.for_mode(
            MODE_RTTY,
            baud=self._config.baud_rate,
            shift_hz=self._config.frequency_shift,
        )


class MorseEncoder(_TextEncoder):
    """
    Morse code encoder using OOK (On-Off Keying).

    Standard timing (PARIS): dit = 1 unit, dah = 3 units, one unit between
    the elements of a letter, three between letters, seven between words.
    """

    MODE = MODE_CW

    #: Character -> Morse code (letters, digits, punctuation).
    MORSE_CODE: Dict[str, str] = dict(MORSE_CODES)

    def __init__(self, config: EncoderConfig, wpm: int = 20):
        """
        Initialize Morse encoder.

        Args:
            config: Encoder configuration
            wpm: Words per minute (default 20)
        """
        _check_range(wpm, WPM_RANGE, "The Morse speed", "WPM")
        super().__init__(config)
        self._wpm = wpm
        # PARIS standard: 1 word = 50 units
        self._dit_duration = 1.2 / wpm  # seconds

    def _settings(self) -> TextSettings:
        return TextSettings.for_mode(MODE_CW, wpm=self._wpm)


class ASCIIEncoder(_TextEncoder):
    """
    Simple ASCII encoder using FSK modulation.

    Each character is 8 bits (standard ASCII) with start/stop bits (8N1).
    """

    MODE = MODE_ASCII

    def __init__(self, config: EncoderConfig):
        """Initialize ASCII encoder."""
        if not config.frequency_shift:
            config.frequency_shift = 1000
        if not config.baud_rate:
            config.baud_rate = ASCII_BAUD
        _check_range(config.baud_rate, BAUD_RANGE, "The ASCII baud rate", "Bd")
        _check_range(config.frequency_shift, SHIFT_RANGE, "The ASCII shift", "Hz")
        super().__init__(config)

    def _settings(self) -> TextSettings:
        assert self._config.frequency_shift is not None
        return TextSettings.for_mode(
            MODE_ASCII,
            baud=self._config.baud_rate,
            shift_hz=self._config.frequency_shift,
        )

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to FSK signal: idle mark, the 8N1 frames of its ASCII
        bytes, idle mark. Characters outside ASCII are left out and listed
        in :attr:`skipped`.

        Args:
            text: Text to encode

        Returns:
            Complex I/Q samples
        """
        return super().encode_text(text)

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """
        Encode bytes to FSK signal with framing (no idle before or after).

        Each byte:
        - 1 start bit (0)
        - 8 data bits (LSB first)
        - 1 stop bit (1)

        Args:
            data: Bytes to encode

        Returns:
            Complex I/Q samples
        """
        bits = []

        for byte in data:
            # Start bit
            bits.append(0)

            # Data bits (LSB first)
            for i in range(8):
                bits.append((byte >> i) & 1)

            # Stop bit
            bits.append(1)

        bit_array = np.array(bits, dtype=np.uint8)

        # Modulate using FSK
        assert self._config.frequency_shift is not None
        mark_freq = self._carrier_freq + self._config.frequency_shift / 2
        space_freq = self._carrier_freq - self._config.frequency_shift / 2

        return self.bits_to_fsk(bit_array, mark_freq, space_freq)


class PSK31Encoder(_TextEncoder):
    """
    PSK31 encoder - popular digital mode for amateur radio.

    Binary PSK at 31.25 baud with G3PLX's Varicode (all 128 ASCII
    characters, case-sensitive); phase reversals are cosine-shaped.
    """

    MODE = MODE_PSK31

    #: Character -> Varicode (the full table).
    VARICODE: Dict[str, str] = dict(VARICODE)

    def __init__(self, config: EncoderConfig):
        """Initialize PSK31 encoder."""
        config.baud_rate = PSK31_BAUD  # Fixed baud rate for PSK31
        config.modulation = ModulationType.PSK
        super().__init__(config)

    def _settings(self) -> TextSettings:
        return TextSettings.for_mode(MODE_PSK31)

"""
Specific protocol encoder implementations.

Provides ready-to-use encoders for common protocols.
"""

import numpy as np
from .encoder import ProtocolEncoder, EncoderConfig, ModulationType


class RTTYEncoder(ProtocolEncoder):
    """
    Radio Teletype (RTTY) encoder.

    Uses FSK modulation with Baudot code (5-bit).
    Standard shift is 170 Hz for amateur radio.
    """

    # Baudot code table (letters mode)
    BAUDOT_LETTERS = {
        'A': 0b00011, 'B': 0b11001, 'C': 0b01110, 'D': 0b01001,
        'E': 0b00001, 'F': 0b01101, 'G': 0b11010, 'H': 0b10100,
        'I': 0b00110, 'J': 0b01011, 'K': 0b01111, 'L': 0b10010,
        'M': 0b11100, 'N': 0b01100, 'O': 0b11000, 'P': 0b10110,
        'Q': 0b10111, 'R': 0b01010, 'S': 0b00101, 'T': 0b10000,
        'U': 0b00111, 'V': 0b11110, 'W': 0b10011, 'X': 0b11101,
        'Y': 0b10101, 'Z': 0b10001, ' ': 0b00100,
    }

    def __init__(self, config: EncoderConfig):
        """Initialize RTTY encoder with 45.45 baud default."""
        if config.baud_rate == 0:
            config.baud_rate = 45.45  # Standard RTTY baud rate
        if not config.frequency_shift:
            config.frequency_shift = 170  # Standard shift
        super().__init__(config)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to RTTY FSK signal.

        Args:
            text: Text to encode (uppercase recommended)

        Returns:
            Complex I/Q samples
        """
        text = text.upper()
        bits = []

        for char in text:
            if char in self.BAUDOT_LETTERS:
                # Start bit (0)
                bits.append(0)

                # 5 data bits (LSB first)
                code = self.BAUDOT_LETTERS[char]
                for i in range(5):
                    bits.append((code >> i) & 1)

                # Stop bit (1)
                bits.append(1)

        bits = np.array(bits, dtype=np.uint8)

        # Modulate using FSK
        mark_freq = self._carrier_freq + self._config.frequency_shift / 2
        space_freq = self._carrier_freq - self._config.frequency_shift / 2

        return self.bits_to_fsk(bits, mark_freq, space_freq)

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Encode bytes (converts to text first)."""
        try:
            text = data.decode('ascii')
            return self.encode_text(text)
        except UnicodeDecodeError:
            return np.array([], dtype=np.complex64)


class MorseEncoder(ProtocolEncoder):
    """
    Morse code encoder using OOK (On-Off Keying).

    Standard timing: Dit = 1 unit, Dah = 3 units.
    """

    MORSE_CODE = {
        'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
        'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
        'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
        'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
        'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
        'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
        'Y': '-.--',  'Z': '--..',
        '0': '-----', '1': '.----', '2': '..---', '3': '...--',
        '4': '....-', '5': '.....', '6': '-....', '7': '--...',
        '8': '---..', '9': '----.',
        ' ': ' ',     '/': '-..-.'
    }

    def __init__(self, config: EncoderConfig, wpm: int = 20):
        """
        Initialize Morse encoder.

        Args:
            config: Encoder configuration
            wpm: Words per minute (default 20)
        """
        super().__init__(config)
        self._wpm = wpm
        # PARIS standard: 1 word = 50 units
        self._dit_duration = 1.2 / wpm  # seconds

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to Morse code.

        Args:
            text: Text to encode

        Returns:
            Complex I/Q samples (OOK modulated)
        """
        text = text.upper()
        morse_pattern = []

        for char in text:
            if char in self.MORSE_CODE:
                morse_pattern.append(self.MORSE_CODE[char])

        # Convert morse pattern to timing units
        signal_parts = []
        t_carrier = np.arange(
            int(self._dit_duration * self._sample_rate)
        ) / self._sample_rate
        carrier = self._config.amplitude * np.exp(
            2j * np.pi * self._carrier_freq * t_carrier
        )

        silence = np.zeros_like(carrier)

        for morse_char in morse_pattern:
            for symbol in morse_char:
                if symbol == '.':
                    # Dit (1 unit)
                    signal_parts.append(carrier)
                    signal_parts.append(silence)  # Inter-symbol gap
                elif symbol == '-':
                    # Dah (3 units)
                    signal_parts.append(np.tile(carrier, 3))
                    signal_parts.append(silence)  # Inter-symbol gap
                elif symbol == ' ':
                    # Word space (7 units)
                    signal_parts.append(np.tile(silence, 7))

            # Inter-character gap (3 units total, 1 already added)
            signal_parts.append(np.tile(silence, 2))

        return np.concatenate(signal_parts).astype(np.complex64)

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Encode bytes as Morse (converts to text)."""
        try:
            text = data.decode('ascii')
            return self.encode_text(text)
        except UnicodeDecodeError:
            return np.array([], dtype=np.complex64)


class ASCIIEncoder(ProtocolEncoder):
    """
    Simple ASCII encoder using FSK modulation.

    Each character is 8 bits (standard ASCII) with start/stop bits.
    """

    def __init__(self, config: EncoderConfig):
        """Initialize ASCII encoder."""
        if not config.frequency_shift:
            config.frequency_shift = 1000
        super().__init__(config)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to FSK signal.

        Args:
            text: Text to encode

        Returns:
            Complex I/Q samples
        """
        return self.encode_bytes(text.encode('ascii'))

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """
        Encode bytes to FSK signal with framing.

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

        bits = np.array(bits, dtype=np.uint8)

        # Modulate using FSK
        mark_freq = self._carrier_freq + self._config.frequency_shift / 2
        space_freq = self._carrier_freq - self._config.frequency_shift / 2

        return self.bits_to_fsk(bits, mark_freq, space_freq)


class PSK31Encoder(ProtocolEncoder):
    """
    PSK31 encoder - popular digital mode for amateur radio.

    Uses Binary Phase Shift Keying at 31.25 baud.
    """

    # Varicode table (variable-length encoding)
    VARICODE = {
        ' ': '1', 'e': '11', 't': '101', 'o': '111', 'a': '1011',
        'n': '1101', 'i': '1111', 's': '10101', 'r': '10111',
        'h': '11011', 'l': '11101', 'd': '11111', 'c': '101011',
        # Add more as needed...
    }

    def __init__(self, config: EncoderConfig):
        """Initialize PSK31 encoder."""
        config.baud_rate = 31.25  # Fixed baud rate for PSK31
        config.modulation = ModulationType.PSK
        super().__init__(config)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using PSK31 varicode.

        Args:
            text: Text to encode

        Returns:
            Complex I/Q samples
        """
        text = text.lower()
        bits_str = ''

        for char in text:
            if char in self.VARICODE:
                bits_str += self.VARICODE[char] + '00'  # Double zero separator

        bits = np.array([int(b) for b in bits_str], dtype=np.uint8)

        # PSK modulation
        samples_per_bit = int(self._sample_rate / self._baud_rate)
        total_samples = len(bits) * samples_per_bit

        signal = np.zeros(total_samples, dtype=np.complex64)
        t = np.arange(samples_per_bit) / self._sample_rate

        phase = 0.0
        for i, bit in enumerate(bits):
            # Phase shift by π for bit transitions
            if bit == 1:
                phase = (phase + np.pi) % (2 * np.pi)

            carrier_phase = 2 * np.pi * self._carrier_freq * t + phase

            start_idx = i * samples_per_bit
            end_idx = start_idx + samples_per_bit

            signal[start_idx:end_idx] = (
                self._config.amplitude * np.exp(1j * carrier_phase)
            )

        return signal

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Encode bytes (converts to text)."""
        try:
            text = data.decode('ascii')
            return self.encode_text(text)
        except UnicodeDecodeError:
            return np.array([], dtype=np.complex64)

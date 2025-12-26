"""
Protocol encoder framework for converting data to transmittable signals.

Provides base classes and utilities for encoding text and data
into various radio protocol formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np


class ModulationType(Enum):
    """Modulation types for encoding."""
    FSK = "fsk"          # Frequency Shift Keying
    ASK = "ask"          # Amplitude Shift Keying
    PSK = "psk"          # Phase Shift Keying
    OOK = "ook"          # On-Off Keying
    AFSK = "afsk"        # Audio FSK
    MSK = "msk"          # Minimum Shift Keying


@dataclass
class EncoderConfig:
    """Configuration for protocol encoder."""
    sample_rate: float
    carrier_freq: float
    baud_rate: float
    modulation: ModulationType
    amplitude: float = 1.0
    frequency_shift: Optional[float] = None  # For FSK
    phase_shift: Optional[float] = None      # For PSK


class ProtocolEncoder(ABC):
    """
    Abstract base class for protocol encoders.

    Encodes text/data into modulated I/Q samples for transmission.
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize encoder.

        Args:
            config: Encoder configuration
        """
        self._config = config
        self._sample_rate = config.sample_rate
        self._carrier_freq = config.carrier_freq
        self._baud_rate = config.baud_rate

    @property
    def config(self) -> EncoderConfig:
        """Get encoder configuration."""
        return self._config

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to I/Q samples.

        Args:
            text: Text to encode

        Returns:
            Complex I/Q samples ready for transmission
        """
        pass

    @abstractmethod
    def encode_bytes(self, data: bytes) -> np.ndarray:
        """
        Encode raw bytes to I/Q samples.

        Args:
            data: Binary data to encode

        Returns:
            Complex I/Q samples
        """
        pass

    def text_to_bits(self, text: str, encoding: str = 'ascii') -> np.ndarray:
        """
        Convert text to bit array.

        Args:
            text: Text to convert
            encoding: Character encoding to use

        Returns:
            Bit array
        """
        byte_data = text.encode(encoding)
        bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        return bits

    def bits_to_fsk(
        self,
        bits: np.ndarray,
        mark_freq: float,
        space_freq: float
    ) -> np.ndarray:
        """
        Modulate bits using FSK.

        Args:
            bits: Bit array
            mark_freq: Frequency for '1' bits
            space_freq: Frequency for '0' bits

        Returns:
            Complex FSK modulated signal
        """
        samples_per_bit = int(self._sample_rate / self._baud_rate)
        total_samples = len(bits) * samples_per_bit

        signal = np.zeros(total_samples, dtype=np.complex64)
        t = np.arange(samples_per_bit) / self._sample_rate

        for i, bit in enumerate(bits):
            freq = mark_freq if bit else space_freq
            phase = 2 * np.pi * freq * t

            start_idx = i * samples_per_bit
            end_idx = start_idx + samples_per_bit

            signal[start_idx:end_idx] = (
                self._config.amplitude * np.exp(1j * phase)
            )

        return signal

    def bits_to_ask(self, bits: np.ndarray) -> np.ndarray:
        """
        Modulate bits using ASK (Amplitude Shift Keying).

        Args:
            bits: Bit array

        Returns:
            Complex ASK modulated signal
        """
        samples_per_bit = int(self._sample_rate / self._baud_rate)
        total_samples = len(bits) * samples_per_bit

        signal = np.zeros(total_samples, dtype=np.complex64)
        t = np.arange(samples_per_bit) / self._sample_rate
        carrier_phase = 2 * np.pi * self._carrier_freq * t
        carrier = np.exp(1j * carrier_phase)

        for i, bit in enumerate(bits):
            amplitude = self._config.amplitude if bit else 0.0

            start_idx = i * samples_per_bit
            end_idx = start_idx + samples_per_bit

            signal[start_idx:end_idx] = amplitude * carrier

        return signal

    def add_preamble(
        self,
        signal: np.ndarray,
        preamble_bits: np.ndarray
    ) -> np.ndarray:
        """
        Add preamble to signal.

        Args:
            signal: Original signal
            preamble_bits: Preamble bit pattern

        Returns:
            Signal with preamble prepended
        """
        # Encode preamble using same modulation
        if self._config.modulation == ModulationType.FSK:
            freq_shift = self._config.frequency_shift or 1000
            preamble = self.bits_to_fsk(
                preamble_bits,
                self._carrier_freq + freq_shift / 2,
                self._carrier_freq - freq_shift / 2
            )
        else:
            preamble = self.bits_to_ask(preamble_bits)

        return np.concatenate([preamble, signal])

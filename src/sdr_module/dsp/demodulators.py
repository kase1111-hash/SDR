"""
Signal demodulators for common modulation schemes.

Supports analog and digital modulation types:
- AM, FM, SSB (analog)
- ASK/OOK, FSK, PSK (digital)
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod


class ModulationType(Enum):
    """Modulation types."""
    # Analog
    AM = "am"
    FM = "fm"
    USB = "usb"
    LSB = "lsb"
    CW = "cw"
    # Digital
    OOK = "ook"
    ASK = "ask"
    FSK = "fsk"
    BPSK = "bpsk"
    QPSK = "qpsk"
    GFSK = "gfsk"


class Demodulator(ABC):
    """Abstract base class for demodulators."""

    def __init__(self, sample_rate: float):
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @abstractmethod
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate samples."""
        pass

    def reset(self) -> None:
        """Reset demodulator state."""
        pass


class AMDemodulator(Demodulator):
    """
    AM envelope demodulator.

    Uses envelope detection (magnitude) for AM demodulation.
    """

    def __init__(self, sample_rate: float, dc_block: bool = True):
        super().__init__(sample_rate)
        self._dc_block = dc_block
        self._dc_avg = 0.0
        self._dc_alpha = 0.001

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate AM signal."""
        # Envelope detection
        envelope = np.abs(samples)

        # DC blocking (remove carrier component)
        if self._dc_block:
            output = np.zeros_like(envelope)
            for i, s in enumerate(envelope):
                self._dc_avg = self._dc_alpha * s + (1 - self._dc_alpha) * self._dc_avg
                output[i] = s - self._dc_avg
            return output

        return envelope

    def reset(self) -> None:
        self._dc_avg = 0.0


class FMDemodulator(Demodulator):
    """
    FM demodulator using quadrature detection.

    Computes instantaneous frequency from phase differences.
    """

    def __init__(self, sample_rate: float, max_deviation: float = 75e3):
        super().__init__(sample_rate)
        self._max_deviation = max_deviation
        self._last_sample = 0.0 + 0.0j

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FM signal using quadrature detection."""
        # Prepend last sample for continuity
        extended = np.concatenate([[self._last_sample], samples])
        self._last_sample = samples[-1]

        # Quadrature demodulation
        # d(angle)/dt = Im(conj(x[n-1]) * x[n]) / |x[n-1]|^2
        delayed = extended[:-1]
        current = extended[1:]

        # Compute phase difference
        product = np.conj(delayed) * current
        phase_diff = np.angle(product)

        # Normalize by max deviation
        demod = phase_diff * (self._sample_rate / (2 * np.pi * self._max_deviation))

        return demod

    def reset(self) -> None:
        self._last_sample = 0.0 + 0.0j


class SSBDemodulator(Demodulator):
    """
    SSB (Single Sideband) demodulator.

    Supports USB (upper sideband) and LSB (lower sideband).
    """

    def __init__(self, sample_rate: float, mode: str = "usb"):
        super().__init__(sample_rate)
        self._mode = mode.lower()

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate SSB signal."""
        if self._mode == "lsb":
            # For LSB, conjugate to flip spectrum
            samples = np.conj(samples)

        # Extract real part (product detection with carrier)
        return samples.real


class OOKDemodulator(Demodulator):
    """
    OOK (On-Off Keying) demodulator.

    Simple threshold-based digital demodulation.
    """

    def __init__(self, sample_rate: float, threshold: float = 0.5):
        super().__init__(sample_rate)
        self._threshold = threshold
        self._auto_threshold = True

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate OOK signal to bits."""
        envelope = np.abs(samples)

        if self._auto_threshold:
            # Automatic threshold based on signal statistics
            threshold = (np.max(envelope) + np.min(envelope)) / 2
        else:
            threshold = self._threshold * np.max(envelope)

        return (envelope > threshold).astype(np.float32)

    def set_threshold(self, threshold: float, auto: bool = False) -> None:
        """Set demodulation threshold."""
        self._threshold = threshold
        self._auto_threshold = auto


class FSKDemodulator(Demodulator):
    """
    FSK (Frequency Shift Keying) demodulator.

    Uses quadrature detection to recover frequency shifts.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        deviation: float = 2400
    ):
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._deviation = deviation
        self._fm_demod = FMDemodulator(sample_rate, deviation)

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FSK signal."""
        # First, FM demodulate
        freq = self._fm_demod.demodulate(samples)

        # The output represents frequency deviation
        # Positive = mark, Negative = space (for standard FSK)
        return freq

    def demodulate_bits(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate FSK to bits with symbol timing."""
        freq = self.demodulate(samples)

        # Simple slicing at threshold
        bits = (freq > 0).astype(np.float32)

        return bits

    def reset(self) -> None:
        self._fm_demod.reset()


class PSKDemodulator(Demodulator):
    """
    PSK (Phase Shift Keying) demodulator.

    Supports BPSK, QPSK, and higher-order PSK.
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        order: int = 2  # 2=BPSK, 4=QPSK
    ):
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._order = order
        self._last_phase = 0.0

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate PSK signal to symbols."""
        # Extract phase
        phase = np.angle(samples)

        if self._order == 2:  # BPSK
            # Map to 0 or 1
            symbols = (phase > 0).astype(np.float32)
        elif self._order == 4:  # QPSK
            # Map to 0, 1, 2, 3
            symbols = np.floor((phase + np.pi) / (np.pi / 2)) % 4
        else:
            # General M-PSK
            symbols = np.floor((phase + np.pi) / (2 * np.pi / self._order)) % self._order

        return symbols

    def get_constellation(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get I/Q constellation points."""
        return samples.real, samples.imag


# Morse code lookup table
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6',
    '--...': '7', '---..': '8', '----.': '9', '.-.-.-': '.',
    '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
    '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&',
    '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+',
    '-....-': '-', '..--.-': '_', '.-..-.': '"', '...-..-': '$',
    '.--.-.': '@', '...---...': 'SOS',
}


class CWDemodulator(Demodulator):
    """
    CW (Continuous Wave / Morse Code) demodulator.

    Generates a BFO (Beat Frequency Oscillator) tone to make the
    carrier audible, detects on/off keying, and optionally decodes
    to text.

    Features:
    - Configurable BFO offset frequency
    - Automatic gain control
    - On/off keying detection
    - Timing analysis for dots/dashes
    - Morse code to text decoding
    """

    def __init__(
        self,
        sample_rate: float,
        bfo_freq: float = 700.0,
        bandwidth: float = 500.0,
        wpm: float = 15.0
    ):
        """
        Initialize CW demodulator.

        Args:
            sample_rate: Sample rate in Hz
            bfo_freq: BFO frequency offset in Hz (typical 400-1000 Hz)
            bandwidth: Filter bandwidth in Hz
            wpm: Expected words per minute for timing
        """
        super().__init__(sample_rate)
        self._bfo_freq = bfo_freq
        self._bandwidth = bandwidth
        self._wpm = wpm

        # BFO state
        self._bfo_phase = 0.0

        # AGC state
        self._agc_gain = 1.0
        self._agc_alpha = 0.001

        # Envelope detector state
        self._envelope_avg = 0.0
        self._envelope_alpha = 0.01

        # Keying detector state
        self._threshold = 0.5
        self._key_state = False
        self._key_time = 0.0

        # Timing parameters (based on WPM)
        # PARIS standard: 50 units per word
        # 1 WPM = 60 seconds / 50 = 1.2 seconds per word
        # 1 unit = 1.2 / WPM seconds
        self._unit_time = 1.2 / wpm
        self._dot_time = self._unit_time
        self._dash_time = self._unit_time * 3
        self._element_gap = self._unit_time
        self._letter_gap = self._unit_time * 3
        self._word_gap = self._unit_time * 7

        # Morse decoder state
        self._current_element = ""
        self._current_letter = ""
        self._decoded_text = ""
        self._last_key_time = 0.0
        self._sample_count = 0

    @property
    def bfo_frequency(self) -> float:
        """Get BFO frequency."""
        return self._bfo_freq

    @bfo_frequency.setter
    def bfo_frequency(self, freq: float) -> None:
        """Set BFO frequency."""
        self._bfo_freq = freq

    @property
    def wpm(self) -> float:
        """Get words per minute."""
        return self._wpm

    @wpm.setter
    def wpm(self, wpm: float) -> None:
        """Set words per minute and update timing."""
        self._wpm = wpm
        self._unit_time = 1.2 / wpm
        self._dot_time = self._unit_time
        self._dash_time = self._unit_time * 3
        self._element_gap = self._unit_time
        self._letter_gap = self._unit_time * 3
        self._word_gap = self._unit_time * 7

    @property
    def decoded_text(self) -> str:
        """Get decoded Morse text."""
        return self._decoded_text

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate CW signal to audio.

        Mixes with BFO to produce audible tone when carrier is present.

        Args:
            samples: Complex I/Q samples

        Returns:
            Audio output (real-valued)
        """
        n = len(samples)

        # Generate BFO (local oscillator)
        t = np.arange(n) / self._sample_rate
        phase_increment = 2 * np.pi * self._bfo_freq / self._sample_rate
        bfo_phase = self._bfo_phase + np.cumsum(np.ones(n) * phase_increment)
        self._bfo_phase = bfo_phase[-1] % (2 * np.pi)

        bfo = np.exp(1j * bfo_phase)

        # Mix signal with BFO
        mixed = samples * bfo

        # Extract audio (real part)
        audio = np.real(mixed).astype(np.float32)

        # Apply simple AGC
        peak = np.max(np.abs(audio))
        if peak > 0.01:
            target = 0.5
            desired_gain = target / peak
            self._agc_gain = self._agc_alpha * desired_gain + \
                            (1 - self._agc_alpha) * self._agc_gain
        audio = audio * self._agc_gain

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def detect_keying(self, samples: np.ndarray) -> np.ndarray:
        """
        Detect on/off keying from samples.

        Args:
            samples: Complex I/Q samples or audio

        Returns:
            Binary keying signal (1 = key down, 0 = key up)
        """
        # Get envelope
        if np.iscomplexobj(samples):
            envelope = np.abs(samples)
        else:
            # Audio - use absolute value with smoothing
            envelope = np.abs(samples)

        # Smooth envelope
        smoothed = np.zeros_like(envelope)
        avg = self._envelope_avg
        for i, s in enumerate(envelope):
            avg = self._envelope_alpha * s + (1 - self._envelope_alpha) * avg
            smoothed[i] = avg
        self._envelope_avg = avg

        # Auto-threshold
        env_max = np.max(smoothed)
        env_min = np.min(smoothed)
        threshold = (env_max + env_min) / 2

        # Apply hysteresis
        keying = np.zeros(len(samples), dtype=np.float32)
        for i, s in enumerate(smoothed):
            if self._key_state:
                # Key is down - go up when below threshold * 0.7
                if s < threshold * 0.7:
                    self._key_state = False
            else:
                # Key is up - go down when above threshold * 1.3
                if s > threshold * 1.3:
                    self._key_state = True
            keying[i] = 1.0 if self._key_state else 0.0

        return keying

    def analyze_timing(self, keying: np.ndarray) -> list:
        """
        Analyze keying timing to detect dots and dashes.

        Args:
            keying: Binary keying signal

        Returns:
            List of (element, duration) tuples
        """
        elements = []

        # Find transitions
        diff = np.diff(np.concatenate([[0], keying, [0]]))
        key_down = np.where(diff > 0)[0]
        key_up = np.where(diff < 0)[0]

        for start, end in zip(key_down, key_up):
            duration_samples = end - start
            duration_sec = duration_samples / self._sample_rate

            # Classify as dot or dash
            if duration_sec < self._dash_time * 0.6:
                element = '.'
            else:
                element = '-'

            elements.append((element, duration_sec))

        return elements

    def decode_morse(self, keying: np.ndarray) -> str:
        """
        Decode Morse code from keying signal.

        Args:
            keying: Binary keying signal

        Returns:
            Decoded text
        """
        self._sample_count += len(keying)
        current_time = self._sample_count / self._sample_rate

        # Find transitions
        diff = np.diff(np.concatenate([[0], keying, [0]]))
        key_down = np.where(diff > 0)[0]
        key_up = np.where(diff < 0)[0]

        decoded = ""

        for i, (start, end) in enumerate(zip(key_down, key_up)):
            # Check gap before this element
            if i == 0 and self._last_key_time > 0:
                gap = start / self._sample_rate + (current_time - len(keying) / self._sample_rate) - self._last_key_time
            elif i > 0:
                gap = (start - key_up[i-1]) / self._sample_rate
            else:
                gap = 0

            # Handle gaps
            if gap > self._word_gap * 0.6:
                # Word gap - decode current letter and add space
                if self._current_letter:
                    char = MORSE_CODE.get(self._current_letter, '?')
                    decoded += char + ' '
                    self._current_letter = ""
            elif gap > self._letter_gap * 0.6:
                # Letter gap - decode current letter
                if self._current_letter:
                    char = MORSE_CODE.get(self._current_letter, '?')
                    decoded += char
                    self._current_letter = ""

            # Classify element
            duration_samples = end - start
            duration_sec = duration_samples / self._sample_rate

            if duration_sec < self._dash_time * 0.6:
                self._current_letter += '.'
            else:
                self._current_letter += '-'

            self._last_key_time = current_time - (len(keying) - end) / self._sample_rate

        self._decoded_text += decoded
        return decoded

    def estimate_wpm(self, keying: np.ndarray) -> float:
        """
        Estimate WPM from keying signal.

        Args:
            keying: Binary keying signal

        Returns:
            Estimated words per minute
        """
        elements = self.analyze_timing(keying)

        if len(elements) < 3:
            return self._wpm  # Not enough data

        # Find shortest element (assumed to be dot)
        durations = [d for _, d in elements]
        dot_estimate = np.percentile(durations, 25)

        if dot_estimate > 0:
            # WPM = 1.2 / unit_time, unit_time = dot_time
            estimated_wpm = 1.2 / dot_estimate
            return max(5, min(50, estimated_wpm))  # Clamp to reasonable range

        return self._wpm

    def reset(self) -> None:
        """Reset demodulator state."""
        self._bfo_phase = 0.0
        self._agc_gain = 1.0
        self._envelope_avg = 0.0
        self._key_state = False
        self._current_letter = ""
        self._decoded_text = ""
        self._last_key_time = 0.0
        self._sample_count = 0

    def clear_decoded(self) -> None:
        """Clear decoded text buffer."""
        self._decoded_text = ""


def create_demodulator(
    mod_type: ModulationType,
    sample_rate: float,
    **kwargs
) -> Demodulator:
    """
    Factory function to create demodulators.

    Args:
        mod_type: Modulation type
        sample_rate: Sample rate in Hz
        **kwargs: Additional demodulator-specific parameters

    Returns:
        Configured demodulator instance
    """
    if mod_type == ModulationType.AM:
        return AMDemodulator(sample_rate, **kwargs)
    elif mod_type == ModulationType.FM:
        return FMDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.USB, ModulationType.LSB):
        return SSBDemodulator(sample_rate, mod_type.value)
    elif mod_type in (ModulationType.OOK, ModulationType.ASK):
        return OOKDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.FSK, ModulationType.GFSK):
        return FSKDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.BPSK, ModulationType.QPSK):
        order = 2 if mod_type == ModulationType.BPSK else 4
        return PSKDemodulator(sample_rate, order=order, **kwargs)
    elif mod_type == ModulationType.CW:
        return CWDemodulator(sample_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

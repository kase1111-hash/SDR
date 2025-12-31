"""
Callsign identification module for HAM radio compliance.

Provides automatic callsign identification to comply with amateur radio
regulations which require operators to identify:
- At the beginning of transmission
- At least every 10 minutes during transmission
- At the end of transmission

Supports multiple identification modes:
- CW (Morse code)
- Voice (pre-recorded or TTS)
- Digital modes (PSK31, RTTY)
"""

from __future__ import annotations

import time
import threading
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np

logger = logging.getLogger(__name__)


class IdentificationMode(Enum):
    """Callsign identification mode."""
    CW = auto()         # Morse code
    VOICE = auto()      # Pre-recorded or TTS
    PSK31 = auto()      # PSK31 digital mode
    RTTY = auto()       # Radio teletype


@dataclass
class CallsignConfig:
    """Configuration for callsign identification."""
    callsign: str = ""
    enabled: bool = True
    mode: IdentificationMode = IdentificationMode.CW
    interval_seconds: int = 600  # 10 minutes (FCC requirement)
    id_at_start: bool = True
    id_at_end: bool = True
    cw_wpm: int = 20  # Words per minute for CW
    cw_frequency: float = 700.0  # Sidetone frequency in Hz
    voice_file: Optional[str] = None  # Path to voice ID file
    sample_rate: float = 48000.0


@dataclass
class IdentificationState:
    """Current state of the identification system."""
    is_transmitting: bool = False
    last_id_time: float = 0.0
    transmission_start_time: float = 0.0
    id_count: int = 0
    pending_id: bool = False


class MorseEncoder:
    """Morse code encoder for CW identification."""

    # International Morse Code
    MORSE_CODE = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
        'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
        'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
        'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
        'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....', '6': '-....',
        '7': '--...', '8': '---..', '9': '----.', '/': '-..-.',
        ' ': ' ', '.': '.-.-.-', ',': '--..--', '?': '..--..',
    }

    def __init__(self, wpm: int = 20, frequency: float = 700.0,
                 sample_rate: float = 48000.0):
        """
        Initialize Morse encoder.

        Args:
            wpm: Words per minute
            frequency: Tone frequency in Hz
            sample_rate: Audio sample rate in Hz
        """
        self.wpm = wpm
        self.frequency = frequency
        self.sample_rate = sample_rate

        # Timing based on "PARIS" standard (50 dit units per word)
        self.dit_duration = 60.0 / (50 * wpm)  # seconds per dit
        self.dah_duration = 3 * self.dit_duration
        self.intra_char_gap = self.dit_duration
        self.inter_char_gap = 3 * self.dit_duration
        self.word_gap = 7 * self.dit_duration

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to Morse code audio.

        Args:
            text: Text to encode (typically callsign)

        Returns:
            Audio samples as numpy array
        """
        samples = []
        text = text.upper().strip()

        for i, char in enumerate(text):
            if char == ' ':
                # Word gap
                samples.append(self._generate_silence(self.word_gap))
            elif char in self.MORSE_CODE:
                morse = self.MORSE_CODE[char]
                for j, symbol in enumerate(morse):
                    if symbol == '.':
                        samples.append(self._generate_tone(self.dit_duration))
                    elif symbol == '-':
                        samples.append(self._generate_tone(self.dah_duration))

                    # Intra-character gap (except after last symbol)
                    if j < len(morse) - 1:
                        samples.append(self._generate_silence(self.intra_char_gap))

                # Inter-character gap (except after last character)
                if i < len(text) - 1 and text[i + 1] != ' ':
                    samples.append(self._generate_silence(self.inter_char_gap))

        if samples:
            return np.concatenate(samples)
        return np.array([], dtype=np.float32)

    def _generate_tone(self, duration: float) -> np.ndarray:
        """Generate a tone of specified duration."""
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate

        # Generate tone with smooth envelope to prevent clicks
        tone = np.sin(2 * np.pi * self.frequency * t)

        # Apply raised cosine envelope for smooth edges
        ramp_samples = min(int(0.005 * self.sample_rate), n_samples // 4)
        if ramp_samples > 0:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
            tone[:ramp_samples] *= ramp
            tone[-ramp_samples:] *= ramp[::-1]

        return tone.astype(np.float32)

    def _generate_silence(self, duration: float) -> np.ndarray:
        """Generate silence of specified duration."""
        n_samples = int(duration * self.sample_rate)
        return np.zeros(n_samples, dtype=np.float32)


class CallsignIdentifier:
    """
    Automatic callsign identification system.

    Manages periodic identification to comply with amateur radio regulations.

    Usage:
        identifier = CallsignIdentifier()
        identifier.set_callsign("W1AW")
        identifier.start_transmission()

        # During transmission, check if ID is needed
        if identifier.needs_identification():
            audio = identifier.generate_id()
            # Transmit the audio...
            identifier.mark_identified()

        identifier.end_transmission()
    """

    def __init__(self, config: Optional[CallsignConfig] = None):
        """
        Initialize the callsign identifier.

        Args:
            config: Configuration options
        """
        self.config = config or CallsignConfig()
        self.state = IdentificationState()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._on_id_required: Optional[Callable[[], None]] = None

        # Encoders
        self._morse_encoder: Optional[MorseEncoder] = None

        logger.info(f"CallsignIdentifier initialized")

    def set_callsign(self, callsign: str) -> None:
        """
        Set the operator's callsign.

        Args:
            callsign: Amateur radio callsign (e.g., "W1AW", "VE3XYZ")
        """
        with self._lock:
            # Validate callsign format (basic check)
            callsign = callsign.upper().strip()
            if callsign and not self._validate_callsign(callsign):
                logger.warning(f"Callsign '{callsign}' may not be valid format")

            self.config.callsign = callsign
            logger.info(f"Callsign set to: {callsign}")

    def _validate_callsign(self, callsign: str) -> bool:
        """
        Basic validation of callsign format.

        Most callsigns follow pattern: prefix (1-3 chars) + number + suffix (1-3 chars)
        Examples: W1AW, VE3ABC, JA1XYZ, G3ABC
        """
        if not callsign or len(callsign) < 3 or len(callsign) > 10:
            return False

        # Check for at least one letter and one number
        has_letter = any(c.isalpha() for c in callsign)
        has_number = any(c.isdigit() for c in callsign)

        return has_letter and has_number

    def get_callsign(self) -> str:
        """Get the current callsign."""
        return self.config.callsign

    def set_interval(self, seconds: int) -> None:
        """
        Set the identification interval.

        Args:
            seconds: Interval in seconds (default: 600 = 10 minutes)
        """
        with self._lock:
            self.config.interval_seconds = max(60, min(seconds, 600))
            logger.info(f"ID interval set to {self.config.interval_seconds} seconds")

    def set_mode(self, mode: IdentificationMode) -> None:
        """Set the identification mode."""
        with self._lock:
            self.config.mode = mode
            logger.info(f"ID mode set to {mode.name}")

    def set_cw_speed(self, wpm: int) -> None:
        """Set CW speed in words per minute."""
        with self._lock:
            self.config.cw_wpm = max(5, min(wpm, 50))
            self._morse_encoder = None  # Reset encoder

    def set_on_id_required(self, callback: Callable[[], None]) -> None:
        """
        Set callback for when identification is required.

        The callback will be called when periodic ID is needed during transmission.
        """
        self._on_id_required = callback

    def start_transmission(self) -> Optional[np.ndarray]:
        """
        Signal the start of a transmission.

        Returns:
            ID audio samples if id_at_start is enabled, None otherwise
        """
        with self._lock:
            self.state.is_transmitting = True
            self.state.transmission_start_time = time.time()
            self.state.id_count = 0

            logger.info("Transmission started")

            # Start periodic ID timer
            self._start_id_timer()

            # Generate start ID if enabled
            if self.config.id_at_start and self.config.callsign:
                audio = self._generate_id_audio()
                self.state.last_id_time = time.time()
                self.state.id_count += 1
                logger.info("Start ID generated")
                return audio

        return None

    def end_transmission(self) -> Optional[np.ndarray]:
        """
        Signal the end of a transmission.

        Returns:
            ID audio samples if id_at_end is enabled, None otherwise
        """
        with self._lock:
            self._stop_id_timer()

            audio = None
            if self.config.id_at_end and self.config.callsign:
                audio = self._generate_id_audio()
                self.state.id_count += 1
                logger.info("End ID generated")

            self.state.is_transmitting = False
            duration = time.time() - self.state.transmission_start_time
            logger.info(f"Transmission ended. Duration: {duration:.1f}s, IDs sent: {self.state.id_count}")

            return audio

    def needs_identification(self) -> bool:
        """
        Check if identification is currently required.

        Returns:
            True if ID is needed based on timing
        """
        if not self.config.enabled or not self.config.callsign:
            return False

        if not self.state.is_transmitting:
            return False

        with self._lock:
            if self.state.pending_id:
                return True

            elapsed = time.time() - self.state.last_id_time
            return elapsed >= self.config.interval_seconds

    def generate_id(self) -> np.ndarray:
        """
        Generate identification audio.

        Returns:
            Audio samples for the callsign ID
        """
        with self._lock:
            return self._generate_id_audio()

    def mark_identified(self) -> None:
        """Mark that identification was just sent."""
        with self._lock:
            self.state.last_id_time = time.time()
            self.state.pending_id = False
            self.state.id_count += 1
            logger.debug(f"ID marked. Count: {self.state.id_count}")

    def get_time_until_next_id(self) -> float:
        """
        Get seconds until next required identification.

        Returns:
            Seconds until next ID is required
        """
        if not self.state.is_transmitting:
            return float('inf')

        elapsed = time.time() - self.state.last_id_time
        remaining = self.config.interval_seconds - elapsed
        return max(0, remaining)

    def get_status(self) -> dict:
        """Get current identification status."""
        return {
            "callsign": self.config.callsign,
            "enabled": self.config.enabled,
            "mode": self.config.mode.name,
            "interval_seconds": self.config.interval_seconds,
            "is_transmitting": self.state.is_transmitting,
            "id_count": self.state.id_count,
            "time_until_next_id": self.get_time_until_next_id(),
            "needs_id": self.needs_identification(),
        }

    def _generate_id_audio(self) -> np.ndarray:
        """Generate the identification audio based on current mode."""
        callsign = self.config.callsign
        if not callsign:
            return np.array([], dtype=np.float32)

        if self.config.mode == IdentificationMode.CW:
            return self._generate_cw_id(callsign)
        elif self.config.mode == IdentificationMode.VOICE:
            return self._generate_voice_id(callsign)
        elif self.config.mode == IdentificationMode.PSK31:
            return self._generate_psk31_id(callsign)
        elif self.config.mode == IdentificationMode.RTTY:
            return self._generate_rtty_id(callsign)
        else:
            return self._generate_cw_id(callsign)

    def _generate_cw_id(self, callsign: str) -> np.ndarray:
        """Generate CW (Morse code) identification."""
        if self._morse_encoder is None:
            self._morse_encoder = MorseEncoder(
                wpm=self.config.cw_wpm,
                frequency=self.config.cw_frequency,
                sample_rate=self.config.sample_rate
            )

        # Add "DE" prefix for proper amateur radio format
        id_text = f"DE {callsign}"
        return self._morse_encoder.encode(id_text)

    def _generate_voice_id(self, callsign: str) -> np.ndarray:
        """Generate voice identification (placeholder for TTS or pre-recorded)."""
        # For now, generate a simple tone sequence as placeholder
        # In a full implementation, this would use TTS or load a pre-recorded file
        logger.warning("Voice ID not fully implemented, using CW fallback")
        return self._generate_cw_id(callsign)

    def _generate_psk31_id(self, callsign: str) -> np.ndarray:
        """Generate PSK31 identification."""
        # Simplified PSK31 - in full implementation would use proper varicode
        logger.warning("PSK31 ID not fully implemented, using CW fallback")
        return self._generate_cw_id(callsign)

    def _generate_rtty_id(self, callsign: str) -> np.ndarray:
        """Generate RTTY identification."""
        # Simplified RTTY - in full implementation would use proper Baudot
        logger.warning("RTTY ID not fully implemented, using CW fallback")
        return self._generate_cw_id(callsign)

    def _start_id_timer(self) -> None:
        """Start the periodic ID timer."""
        self._stop_id_timer()

        if self.config.enabled and self.config.interval_seconds > 0:
            self._timer = threading.Timer(
                self.config.interval_seconds,
                self._on_timer_expired
            )
            self._timer.daemon = True
            self._timer.start()

    def _stop_id_timer(self) -> None:
        """Stop the periodic ID timer."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _on_timer_expired(self) -> None:
        """Called when the ID timer expires."""
        with self._lock:
            if self.state.is_transmitting:
                self.state.pending_id = True
                logger.info("Periodic ID required")

                if self._on_id_required:
                    self._on_id_required()

                # Restart timer for next interval
                self._start_id_timer()


# Convenience function for quick callsign ID generation
def generate_cw_id(callsign: str, wpm: int = 20,
                   frequency: float = 700.0,
                   sample_rate: float = 48000.0) -> np.ndarray:
    """
    Generate a CW callsign identification.

    Args:
        callsign: The callsign to encode
        wpm: Words per minute (default: 20)
        frequency: Tone frequency in Hz (default: 700)
        sample_rate: Sample rate in Hz (default: 48000)

    Returns:
        Audio samples as numpy array
    """
    encoder = MorseEncoder(wpm=wpm, frequency=frequency, sample_rate=sample_rate)
    return encoder.encode(f"DE {callsign}")


def audio_to_fm_iq(
    audio: np.ndarray,
    audio_sample_rate: float = 48000.0,
    rf_sample_rate: float = 2e6,
    deviation_hz: float = 5000.0,
) -> np.ndarray:
    """
    Convert audio samples to FM-modulated I/Q samples for transmission.

    This function takes baseband audio (e.g., from CW ID generator) and
    produces complex I/Q samples suitable for SDR transmission.

    Args:
        audio: Audio samples (mono, float, -1 to 1 range)
        audio_sample_rate: Sample rate of input audio in Hz (default: 48000)
        rf_sample_rate: Desired output sample rate in Hz (default: 2MHz)
        deviation_hz: FM deviation in Hz (default: 5kHz for narrowband)

    Returns:
        Complex I/Q samples as numpy array (complex64)

    Example:
        >>> audio = generate_cw_id("W1AW")
        >>> iq_samples = audio_to_fm_iq(audio, deviation_hz=2500)  # CW uses ~2.5kHz
        >>> hackrf.write_samples(iq_samples)
    """
    if len(audio) == 0:
        return np.array([], dtype=np.complex64)

    # Normalize audio to -1 to 1 range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Resample audio to RF sample rate if needed
    if audio_sample_rate != rf_sample_rate:
        # Calculate resampling ratio
        ratio = rf_sample_rate / audio_sample_rate
        new_length = int(len(audio) * ratio)

        # Use linear interpolation for resampling
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(new_indices, old_indices, audio)

    # FM modulation: phase = integral of frequency deviation
    # frequency deviation = deviation_hz * audio_signal
    # phase = 2*pi * integral(deviation_hz * audio) / sample_rate

    # Calculate instantaneous phase
    phase_increment = 2 * np.pi * deviation_hz / rf_sample_rate
    phase = np.cumsum(audio) * phase_increment

    # Generate complex I/Q signal: e^(j*phase)
    iq_samples = np.exp(1j * phase).astype(np.complex64)

    return iq_samples


def generate_tx_id(
    callsign: str,
    wpm: int = 20,
    tone_frequency: float = 700.0,
    rf_sample_rate: float = 2e6,
    fm_deviation: float = 2500.0,
) -> np.ndarray:
    """
    Generate a ready-to-transmit FM-modulated CW callsign ID.

    Combines CW audio generation with FM modulation for direct transmission.

    Args:
        callsign: The callsign to encode
        wpm: Words per minute (default: 20)
        tone_frequency: CW tone frequency in Hz (default: 700)
        rf_sample_rate: Output sample rate in Hz (default: 2MHz)
        fm_deviation: FM deviation in Hz (default: 2500 for CW)

    Returns:
        Complex I/Q samples ready for transmission (complex64)
    """
    # Generate CW audio at 48kHz
    audio = generate_cw_id(
        callsign,
        wpm=wpm,
        frequency=tone_frequency,
        sample_rate=48000.0
    )

    # Convert to FM I/Q
    iq = audio_to_fm_iq(
        audio,
        audio_sample_rate=48000.0,
        rf_sample_rate=rf_sample_rate,
        deviation_hz=fm_deviation
    )

    return iq


__all__ = [
    'IdentificationMode',
    'CallsignConfig',
    'CallsignIdentifier',
    'MorseEncoder',
    'generate_cw_id',
    'audio_to_fm_iq',
    'generate_tx_id',
]

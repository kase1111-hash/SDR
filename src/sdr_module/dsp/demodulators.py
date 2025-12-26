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
    MSK = "msk"
    # QAM
    QAM16 = "qam16"
    QAM64 = "qam64"
    QAM256 = "qam256"


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


class GFSKDemodulator(Demodulator):
    """
    GFSK (Gaussian Frequency Shift Keying) demodulator.

    Used in Bluetooth, DECT, and many other protocols.
    Features:
    - Configurable BT (Bandwidth-Time) product
    - Gaussian matched filter
    - Symbol timing recovery
    - Soft and hard decision output
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        deviation: float = 0.5,
        bt: float = 0.5,
        samples_per_symbol: int = 0
    ):
        """
        Initialize GFSK demodulator.

        Args:
            sample_rate: Sample rate in Hz
            symbol_rate: Symbol rate in Hz (baud rate)
            deviation: Modulation index (h = 2*deviation*Ts)
            bt: Bandwidth-Time product (typically 0.3-0.5)
            samples_per_symbol: Samples per symbol (0 = auto-calculate)
        """
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._deviation = deviation
        self._bt = bt

        # Calculate samples per symbol
        if samples_per_symbol <= 0:
            self._sps = int(sample_rate / symbol_rate)
        else:
            self._sps = samples_per_symbol

        # Symbol period
        self._symbol_period = 1.0 / symbol_rate

        # FM demodulator for frequency detection
        self._fm_demod = FMDemodulator(sample_rate, deviation * symbol_rate)

        # Generate Gaussian filter for matched filtering
        self._gaussian_filter = self._generate_gaussian_filter()

        # Timing recovery state
        self._timing_offset = 0.0
        self._timing_alpha = 0.01  # Timing loop gain

        # DC offset tracking
        self._dc_offset = 0.0
        self._dc_alpha = 0.001

        # Last phase for continuous demodulation
        self._last_phase = 0.0

    def _generate_gaussian_filter(self) -> np.ndarray:
        """Generate Gaussian filter impulse response."""
        # Filter spans 4 symbol periods
        span = 4
        n_taps = span * self._sps + 1

        # Time vector
        t = np.arange(n_taps) / self._sps - span / 2

        # Gaussian pulse
        alpha = np.sqrt(np.log(2) / 2) / self._bt
        h = np.sqrt(np.pi) / alpha * np.exp(-(np.pi * t / alpha) ** 2)

        # Normalize
        h = h / np.sum(h)

        return h.astype(np.float32)

    @property
    def symbol_rate(self) -> float:
        """Get symbol rate."""
        return self._symbol_rate

    @property
    def samples_per_symbol(self) -> int:
        """Get samples per symbol."""
        return self._sps

    @property
    def bt_product(self) -> float:
        """Get BT product."""
        return self._bt

    @property
    def gaussian_filter(self) -> np.ndarray:
        """Get Gaussian filter coefficients."""
        return self._gaussian_filter.copy()

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate GFSK signal to frequency deviation.

        Args:
            samples: Complex I/Q samples

        Returns:
            Frequency deviation signal (normalized)
        """
        # FM demodulation (quadrature detector)
        freq = self._fm_demod.demodulate(samples)

        # Apply matched Gaussian filter
        if len(freq) >= len(self._gaussian_filter):
            freq_filtered = np.convolve(freq, self._gaussian_filter, mode='same')
        else:
            freq_filtered = freq

        # DC offset removal
        for i, f in enumerate(freq_filtered):
            self._dc_offset = self._dc_alpha * f + (1 - self._dc_alpha) * self._dc_offset
            freq_filtered[i] = f - self._dc_offset

        return freq_filtered.astype(np.float32)

    def demodulate_bits(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate GFSK signal to bits.

        Args:
            samples: Complex I/Q samples

        Returns:
            Binary bit array
        """
        freq = self.demodulate(samples)

        # Sample at symbol centers
        n_symbols = len(freq) // self._sps
        bits = np.zeros(n_symbols, dtype=np.uint8)

        for i in range(n_symbols):
            # Sample at center of symbol (with timing offset)
            sample_idx = int((i + 0.5) * self._sps + self._timing_offset)
            if 0 <= sample_idx < len(freq):
                bits[i] = 1 if freq[sample_idx] > 0 else 0

        return bits

    def demodulate_soft(self, samples: np.ndarray) -> np.ndarray:
        """
        Soft decision demodulation.

        Args:
            samples: Complex I/Q samples

        Returns:
            Soft bit values (positive = 1, negative = 0)
        """
        freq = self.demodulate(samples)

        # Sample at symbol centers
        n_symbols = len(freq) // self._sps
        soft_bits = np.zeros(n_symbols, dtype=np.float32)

        for i in range(n_symbols):
            sample_idx = int((i + 0.5) * self._sps + self._timing_offset)
            if 0 <= sample_idx < len(freq):
                soft_bits[i] = freq[sample_idx]

        return soft_bits

    def recover_timing(self, samples: np.ndarray) -> float:
        """
        Recover symbol timing from samples.

        Uses early-late gate timing recovery.

        Args:
            samples: Complex I/Q samples

        Returns:
            Estimated timing offset in samples
        """
        freq = self.demodulate(samples)

        # Early-late gate timing error detector
        n_symbols = len(freq) // self._sps
        timing_error = 0.0

        for i in range(1, n_symbols - 1):
            center = int((i + 0.5) * self._sps + self._timing_offset)
            early = center - self._sps // 4
            late = center + self._sps // 4

            if 0 <= early < len(freq) and late < len(freq):
                # Timing error = (late - early) * sign(center)
                decision = 1 if freq[center] > 0 else -1
                error = (abs(freq[late]) - abs(freq[early])) * decision
                timing_error += error

        if n_symbols > 2:
            timing_error /= (n_symbols - 2)

        # Update timing offset
        self._timing_offset += self._timing_alpha * timing_error

        # Keep offset bounded
        self._timing_offset = max(-self._sps / 2, min(self._sps / 2, self._timing_offset))

        return self._timing_offset

    def get_eye_diagram_data(self, samples: np.ndarray, n_symbols: int = 100) -> np.ndarray:
        """
        Get data for eye diagram visualization.

        Args:
            samples: Complex I/Q samples
            n_symbols: Number of symbols to include

        Returns:
            2D array of shape (n_symbols, 2*sps) for eye diagram
        """
        freq = self.demodulate(samples)

        traces = []
        for i in range(min(n_symbols, len(freq) // self._sps - 2)):
            start = i * self._sps
            end = start + 2 * self._sps
            if end <= len(freq):
                traces.append(freq[start:end])

        if traces:
            return np.array(traces, dtype=np.float32)
        return np.array([], dtype=np.float32)

    def estimate_deviation(self, samples: np.ndarray) -> float:
        """
        Estimate the frequency deviation from signal.

        Args:
            samples: Complex I/Q samples

        Returns:
            Estimated deviation in Hz
        """
        freq = self.demodulate(samples)

        # Use peak-to-peak deviation
        if len(freq) > 0:
            deviation = (np.max(freq) - np.min(freq)) / 2
            return deviation * self._symbol_rate
        return 0.0

    def estimate_symbol_rate(self, samples: np.ndarray) -> float:
        """
        Estimate symbol rate from signal.

        Uses autocorrelation to find symbol period.

        Args:
            samples: Complex I/Q samples

        Returns:
            Estimated symbol rate in Hz
        """
        freq = self.demodulate(samples)

        if len(freq) < 2 * self._sps:
            return self._symbol_rate

        # Compute autocorrelation
        freq_centered = freq - np.mean(freq)
        autocorr = np.correlate(freq_centered, freq_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Take positive lags

        # Find first peak after zero lag (symbol period)
        # Skip first few samples
        min_lag = self._sps // 2
        max_lag = self._sps * 2

        if max_lag >= len(autocorr):
            return self._symbol_rate

        search_region = autocorr[min_lag:max_lag]
        if len(search_region) > 0:
            peak_idx = np.argmax(search_region) + min_lag
            if peak_idx > 0:
                return self._sample_rate / peak_idx

        return self._symbol_rate

    def reset(self) -> None:
        """Reset demodulator state."""
        self._fm_demod.reset()
        self._timing_offset = 0.0
        self._dc_offset = 0.0
        self._last_phase = 0.0


class MSKDemodulator(Demodulator):
    """
    MSK (Minimum Shift Keying) demodulator.

    MSK is continuous-phase FSK with modulation index h=0.5,
    providing the minimum bandwidth for orthogonal signaling.
    Used in GSM (as GMSK) and other systems.

    Features:
    - Coherent and non-coherent demodulation
    - Matched filter implementation
    - Symbol timing recovery
    - Phase tracking for coherent detection
    - Can be viewed as offset-QPSK with sinusoidal shaping
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float,
        coherent: bool = True,
        samples_per_symbol: int = 0
    ):
        """
        Initialize MSK demodulator.

        Args:
            sample_rate: Sample rate in Hz
            symbol_rate: Symbol rate in Hz (bit rate)
            coherent: Use coherent demodulation (vs non-coherent)
            samples_per_symbol: Samples per symbol (0 = auto)
        """
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._coherent = coherent

        # MSK has h = 0.5, so freq deviation = symbol_rate / 4
        self._deviation = symbol_rate / 4

        # Calculate samples per symbol
        if samples_per_symbol <= 0:
            self._sps = int(sample_rate / symbol_rate)
        else:
            self._sps = samples_per_symbol

        # Generate matched filters for I and Q arms
        self._i_filter, self._q_filter = self._generate_matched_filters()

        # Phase tracking for coherent demodulation
        self._carrier_phase = 0.0
        self._phase_alpha = 0.01

        # Timing recovery
        self._timing_offset = 0.0
        self._timing_alpha = 0.01

        # For non-coherent: FM demodulator
        self._fm_demod = FMDemodulator(sample_rate, self._deviation * 4)

        # State for continuous operation
        self._last_i = 0.0
        self._last_q = 0.0
        self._bit_polarity = 1  # Alternates for offset QPSK interpretation

    def _generate_matched_filters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate matched filters for MSK.

        MSK can be decomposed into I and Q channels with
        half-sinusoid pulse shaping, offset by half a symbol.
        """
        # Filter length = 2 symbol periods
        n_taps = 2 * self._sps

        t = np.arange(n_taps) / self._sps

        # Half-sinusoid pulses
        # I channel: cos(pi*t/2T) for 0 <= t < 2T
        # Q channel: sin(pi*t/2T) for 0 <= t < 2T
        i_filter = np.cos(np.pi * t / 2)
        q_filter = np.sin(np.pi * t / 2)

        # Normalize
        i_filter = i_filter / np.sqrt(np.sum(i_filter ** 2))
        q_filter = q_filter / np.sqrt(np.sum(q_filter ** 2))

        return i_filter.astype(np.float32), q_filter.astype(np.float32)

    @property
    def symbol_rate(self) -> float:
        """Get symbol rate."""
        return self._symbol_rate

    @property
    def samples_per_symbol(self) -> int:
        """Get samples per symbol."""
        return self._sps

    @property
    def modulation_index(self) -> float:
        """Get modulation index (always 0.5 for MSK)."""
        return 0.5

    @property
    def is_coherent(self) -> bool:
        """Check if using coherent demodulation."""
        return self._coherent

    @property
    def matched_filters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get I and Q matched filter coefficients."""
        return self._i_filter.copy(), self._q_filter.copy()

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate MSK signal.

        Args:
            samples: Complex I/Q samples

        Returns:
            Demodulated signal (frequency for non-coherent,
            matched filter output for coherent)
        """
        if self._coherent:
            return self._demodulate_coherent(samples)
        else:
            return self._demodulate_noncoherent(samples)

    def _demodulate_coherent(self, samples: np.ndarray) -> np.ndarray:
        """Coherent MSK demodulation using matched filters."""
        # Apply carrier phase correction
        phase_correction = np.exp(-1j * self._carrier_phase)
        samples_corrected = samples * phase_correction

        # Separate I and Q
        i_signal = samples_corrected.real
        q_signal = samples_corrected.imag

        # Apply matched filters
        if len(i_signal) >= len(self._i_filter):
            i_filtered = np.convolve(i_signal, self._i_filter, mode='same')
            q_filtered = np.convolve(q_signal, self._q_filter, mode='same')
        else:
            i_filtered = i_signal
            q_filtered = q_signal

        # Combine (MSK can be seen as alternating I/Q decisions)
        output = np.zeros(len(samples), dtype=np.float32)
        for i in range(len(samples)):
            # Alternate between I and Q based on symbol timing
            symbol_idx = i // self._sps
            if symbol_idx % 2 == 0:
                output[i] = i_filtered[i]
            else:
                output[i] = q_filtered[i]

        return output

    def _demodulate_noncoherent(self, samples: np.ndarray) -> np.ndarray:
        """Non-coherent MSK demodulation using FM detection."""
        return self._fm_demod.demodulate(samples).astype(np.float32)

    def demodulate_bits(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate MSK signal to bits.

        Args:
            samples: Complex I/Q samples

        Returns:
            Binary bit array
        """
        if self._coherent:
            return self._demodulate_bits_coherent(samples)
        else:
            return self._demodulate_bits_noncoherent(samples)

    def _demodulate_bits_coherent(self, samples: np.ndarray) -> np.ndarray:
        """Coherent bit demodulation."""
        # Apply carrier phase correction
        phase_correction = np.exp(-1j * self._carrier_phase)
        samples_corrected = samples * phase_correction

        i_signal = samples_corrected.real
        q_signal = samples_corrected.imag

        # Apply matched filters
        if len(i_signal) >= len(self._i_filter):
            i_filtered = np.convolve(i_signal, self._i_filter, mode='same')
            q_filtered = np.convolve(q_signal, self._q_filter, mode='same')
        else:
            i_filtered = i_signal
            q_filtered = q_signal

        # Sample at symbol boundaries
        # MSK: I symbols at even boundaries, Q symbols at odd (offset by T/2)
        n_symbols = len(samples) // self._sps
        bits = np.zeros(n_symbols, dtype=np.uint8)

        for i in range(n_symbols):
            sample_idx = int((i + 0.5) * self._sps + self._timing_offset)
            if 0 <= sample_idx < len(i_filtered):
                if i % 2 == 0:
                    bits[i] = 1 if i_filtered[sample_idx] > 0 else 0
                else:
                    bits[i] = 1 if q_filtered[sample_idx] > 0 else 0

        return bits

    def _demodulate_bits_noncoherent(self, samples: np.ndarray) -> np.ndarray:
        """Non-coherent bit demodulation."""
        freq = self._fm_demod.demodulate(samples)

        n_symbols = len(freq) // self._sps
        bits = np.zeros(n_symbols, dtype=np.uint8)

        for i in range(n_symbols):
            sample_idx = int((i + 0.5) * self._sps + self._timing_offset)
            if 0 <= sample_idx < len(freq):
                bits[i] = 1 if freq[sample_idx] > 0 else 0

        return bits

    def demodulate_soft(self, samples: np.ndarray) -> np.ndarray:
        """
        Soft decision demodulation.

        Args:
            samples: Complex I/Q samples

        Returns:
            Soft bit values
        """
        demod = self.demodulate(samples)

        n_symbols = len(demod) // self._sps
        soft_bits = np.zeros(n_symbols, dtype=np.float32)

        for i in range(n_symbols):
            sample_idx = int((i + 0.5) * self._sps + self._timing_offset)
            if 0 <= sample_idx < len(demod):
                soft_bits[i] = demod[sample_idx]

        return soft_bits

    def track_carrier(self, samples: np.ndarray) -> float:
        """
        Track and correct carrier phase.

        Uses decision-directed phase tracking.

        Args:
            samples: Complex I/Q samples

        Returns:
            Current phase estimate
        """
        # Demodulate to get decisions
        bits = self.demodulate_bits(samples)

        # Reconstruct expected signal phase based on decisions
        # and compare to actual phase
        phase_errors = []

        for i in range(min(len(bits), len(samples) // self._sps)):
            sample_idx = int((i + 0.5) * self._sps)
            if sample_idx < len(samples):
                # Expected phase based on bit decision
                expected_phase = np.pi / 2 if bits[i] else -np.pi / 2

                # Actual phase
                actual_phase = np.angle(samples[sample_idx])

                # Phase error
                error = actual_phase - expected_phase - self._carrier_phase
                # Wrap to [-pi, pi]
                error = np.arctan2(np.sin(error), np.cos(error))
                phase_errors.append(error)

        if phase_errors:
            avg_error = np.mean(phase_errors)
            self._carrier_phase += self._phase_alpha * avg_error
            # Wrap carrier phase
            self._carrier_phase = np.arctan2(
                np.sin(self._carrier_phase),
                np.cos(self._carrier_phase)
            )

        return self._carrier_phase

    def recover_timing(self, samples: np.ndarray) -> float:
        """
        Recover symbol timing.

        Args:
            samples: Complex I/Q samples

        Returns:
            Timing offset in samples
        """
        demod = self.demodulate(samples)

        # Early-late gate
        n_symbols = len(demod) // self._sps
        timing_error = 0.0

        for i in range(1, n_symbols - 1):
            center = int((i + 0.5) * self._sps + self._timing_offset)
            early = center - self._sps // 4
            late = center + self._sps // 4

            if 0 <= early < len(demod) and late < len(demod):
                decision = 1 if demod[center] > 0 else -1
                error = (abs(demod[late]) - abs(demod[early])) * decision
                timing_error += error

        if n_symbols > 2:
            timing_error /= (n_symbols - 2)

        self._timing_offset += self._timing_alpha * timing_error
        self._timing_offset = max(-self._sps / 2, min(self._sps / 2, self._timing_offset))

        return self._timing_offset

    def get_constellation(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get constellation points for visualization.

        MSK constellation alternates between I and Q axes.

        Args:
            samples: Complex I/Q samples

        Returns:
            (I, Q) constellation points
        """
        # Sample at symbol centers
        n_symbols = len(samples) // self._sps
        i_points = []
        q_points = []

        phase_correction = np.exp(-1j * self._carrier_phase)

        for i in range(n_symbols):
            sample_idx = int((i + 0.5) * self._sps + self._timing_offset)
            if 0 <= sample_idx < len(samples):
                point = samples[sample_idx] * phase_correction
                i_points.append(point.real)
                q_points.append(point.imag)

        return np.array(i_points, dtype=np.float32), np.array(q_points, dtype=np.float32)

    def get_eye_diagram_data(self, samples: np.ndarray, n_symbols: int = 100) -> np.ndarray:
        """
        Get data for eye diagram.

        Args:
            samples: Complex I/Q samples
            n_symbols: Number of symbols to include

        Returns:
            2D array for eye diagram plotting
        """
        demod = self.demodulate(samples)

        traces = []
        for i in range(min(n_symbols, len(demod) // self._sps - 2)):
            start = i * self._sps
            end = start + 2 * self._sps
            if end <= len(demod):
                traces.append(demod[start:end])

        if traces:
            return np.array(traces, dtype=np.float32)
        return np.array([], dtype=np.float32)

    def reset(self) -> None:
        """Reset demodulator state."""
        self._carrier_phase = 0.0
        self._timing_offset = 0.0
        self._last_i = 0.0
        self._last_q = 0.0
        self._fm_demod.reset()


class QAMDemodulator(Demodulator):
    """
    QAM (Quadrature Amplitude Modulation) demodulator.

    Supports 16-QAM, 64-QAM, and 256-QAM with:
    - Hard and soft decision output
    - Gray-coded constellation
    - EVM (Error Vector Magnitude) calculation
    - Automatic gain normalization
    """

    def __init__(
        self,
        sample_rate: float,
        symbol_rate: float = 1000.0,
        order: int = 16,
        normalize: bool = True
    ):
        """
        Initialize QAM demodulator.

        Args:
            sample_rate: Sample rate in Hz
            symbol_rate: Symbol rate in Hz
            order: QAM order (16, 64, or 256)
            normalize: Normalize constellation to unit average power
        """
        super().__init__(sample_rate)
        self._symbol_rate = symbol_rate
        self._order = order
        self._normalize = normalize

        # Validate order
        if order not in (16, 64, 256):
            raise ValueError(f"QAM order must be 16, 64, or 256, got {order}")

        # Calculate constellation parameters
        self._bits_per_symbol = int(np.log2(order))
        self._levels = int(np.sqrt(order))  # 4 for 16-QAM, 8 for 64-QAM, 16 for 256-QAM

        # Generate constellation points
        self._constellation = self._generate_constellation()

        # Normalization factor
        self._norm_factor = 1.0
        if normalize:
            avg_power = np.mean(np.abs(self._constellation) ** 2)
            self._norm_factor = 1.0 / np.sqrt(avg_power)
            self._constellation = self._constellation * self._norm_factor

        # EVM tracking
        self._evm_history: list = []
        self._evm_history_max = 100

    def _generate_constellation(self) -> np.ndarray:
        """Generate QAM constellation points with Gray coding."""
        levels = self._levels
        points = []

        # Generate grid points centered at origin
        for i in range(levels):
            for q in range(levels):
                # Map to symmetric levels around 0
                i_val = 2 * i - (levels - 1)
                q_val = 2 * q - (levels - 1)
                points.append(complex(i_val, q_val))

        return np.array(points, dtype=np.complex64)

    @property
    def order(self) -> int:
        """Get QAM order."""
        return self._order

    @property
    def bits_per_symbol(self) -> int:
        """Get bits per symbol."""
        return self._bits_per_symbol

    @property
    def constellation(self) -> np.ndarray:
        """Get constellation points."""
        return self._constellation.copy()

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate QAM signal to symbol indices.

        Args:
            samples: Complex I/Q samples

        Returns:
            Array of symbol indices (0 to order-1)
        """
        # Normalize input if needed
        if self._normalize:
            # Estimate input power and normalize
            input_power = np.mean(np.abs(samples) ** 2)
            if input_power > 0:
                samples = samples / np.sqrt(input_power)

        # Find nearest constellation point for each sample
        symbols = np.zeros(len(samples), dtype=np.int32)
        for i, sample in enumerate(samples):
            distances = np.abs(self._constellation - sample)
            symbols[i] = np.argmin(distances)

        return symbols

    def demodulate_soft(self, samples: np.ndarray) -> np.ndarray:
        """
        Soft decision demodulation.

        Args:
            samples: Complex I/Q samples

        Returns:
            Soft decision values (log-likelihood ratios) for each bit
        """
        n_samples = len(samples)
        n_bits = self._bits_per_symbol
        soft_bits = np.zeros((n_samples, n_bits), dtype=np.float32)

        # Normalize input
        if self._normalize:
            input_power = np.mean(np.abs(samples) ** 2)
            if input_power > 0:
                samples = samples / np.sqrt(input_power)

        # For each sample, compute LLR for each bit
        for i, sample in enumerate(samples):
            distances = np.abs(self._constellation - sample) ** 2

            for bit in range(n_bits):
                # Find constellation points where this bit is 0 or 1
                # Using Gray coding approximation
                bit_mask = 1 << (n_bits - 1 - bit)
                symbols_0 = [j for j in range(self._order) if not (j & bit_mask)]
                symbols_1 = [j for j in range(self._order) if (j & bit_mask)]

                # Min distance to 0 and 1 constellation points
                min_dist_0 = np.min(distances[symbols_0]) if symbols_0 else float('inf')
                min_dist_1 = np.min(distances[symbols_1]) if symbols_1 else float('inf')

                # LLR = log(P(bit=0)/P(bit=1)) ≈ (d1² - d0²) / (2σ²)
                # Using σ² = 1 for normalized constellation
                soft_bits[i, bit] = (min_dist_1 - min_dist_0) / 2.0

        return soft_bits

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Convert symbol indices to bit array.

        Args:
            symbols: Array of symbol indices

        Returns:
            Array of bits
        """
        n_bits = self._bits_per_symbol
        bits = np.zeros(len(symbols) * n_bits, dtype=np.uint8)

        for i, sym in enumerate(symbols):
            for b in range(n_bits):
                bits[i * n_bits + (n_bits - 1 - b)] = (sym >> b) & 1

        return bits

    def calculate_evm(self, samples: np.ndarray) -> float:
        """
        Calculate Error Vector Magnitude.

        Args:
            samples: Complex I/Q samples

        Returns:
            EVM as percentage
        """
        # Normalize input
        if self._normalize:
            input_power = np.mean(np.abs(samples) ** 2)
            if input_power > 0:
                samples = samples / np.sqrt(input_power)

        # Find ideal constellation points
        symbols = self.demodulate(samples * np.sqrt(input_power) if self._normalize else samples)
        ideal_points = self._constellation[symbols]

        # Calculate error vectors
        error_vectors = samples - ideal_points
        error_power = np.mean(np.abs(error_vectors) ** 2)

        # Reference power (average constellation power)
        ref_power = np.mean(np.abs(self._constellation) ** 2)

        # EVM as percentage
        if ref_power > 0:
            evm = 100.0 * np.sqrt(error_power / ref_power)
        else:
            evm = 0.0

        # Track history
        self._evm_history.append(evm)
        if len(self._evm_history) > self._evm_history_max:
            self._evm_history.pop(0)

        return evm

    def get_average_evm(self) -> float:
        """Get average EVM from history."""
        if not self._evm_history:
            return 0.0
        return float(np.mean(self._evm_history))

    def slice_symbols(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slice samples to nearest constellation points.

        Args:
            samples: Complex I/Q samples

        Returns:
            (ideal_points, symbol_indices) tuple
        """
        symbols = self.demodulate(samples)
        ideal_points = self._constellation[symbols]
        return ideal_points, symbols

    def get_constellation_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get constellation I and Q coordinates.

        Returns:
            (I, Q) arrays
        """
        return self._constellation.real, self._constellation.imag

    def get_decision_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get decision boundary lines for constellation.

        Returns:
            (horizontal_lines, vertical_lines) for plotting
        """
        levels = self._levels
        # Decision boundaries are midpoints between levels
        boundaries = np.arange(-(levels - 2), levels, 2)

        if self._normalize:
            # Scale boundaries by same normalization
            boundaries = boundaries * self._norm_factor

        return boundaries, boundaries

    def reset(self) -> None:
        """Reset demodulator state."""
        self._evm_history.clear()


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
    elif mod_type == ModulationType.FSK:
        return FSKDemodulator(sample_rate, **kwargs)
    elif mod_type == ModulationType.GFSK:
        return GFSKDemodulator(sample_rate, **kwargs)
    elif mod_type == ModulationType.MSK:
        return MSKDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.BPSK, ModulationType.QPSK):
        order = 2 if mod_type == ModulationType.BPSK else 4
        return PSKDemodulator(sample_rate, order=order, **kwargs)
    elif mod_type == ModulationType.CW:
        return CWDemodulator(sample_rate, **kwargs)
    elif mod_type in (ModulationType.QAM16, ModulationType.QAM64, ModulationType.QAM256):
        order_map = {ModulationType.QAM16: 16, ModulationType.QAM64: 64, ModulationType.QAM256: 256}
        return QAMDemodulator(sample_rate, order=order_map[mod_type], **kwargs)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

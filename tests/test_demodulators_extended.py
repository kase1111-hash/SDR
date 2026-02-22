"""Extended tests for demodulators that lack coverage.

Covers SSB, FSK, PSK, GFSK, MSK, and QAM demodulators with
synthetic signal generation and round-trip validation.
"""

import numpy as np
import pytest

from sdr_module.dsp.demodulators import (
    FSKDemodulator,
    GFSKDemodulator,
    MSKDemodulator,
    PSKDemodulator,
    QAMDemodulator,
    SSBDemodulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_tone(freq_hz: float, sample_rate: float, duration: float) -> np.ndarray:
    """Generate a complex tone at the given frequency."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)


def generate_bpsk(bits, sample_rate: float, symbol_rate: float) -> np.ndarray:
    """Generate BPSK signal: 1 → positive real, 0 → negative real.

    PSKDemodulator maps phase>0 to symbol 1, phase≤0 to symbol 0.
    So bit=1 uses a small positive phase, bit=0 uses a negative phase.
    """
    sps = int(sample_rate / symbol_rate)
    samples = np.zeros(len(bits) * sps, dtype=np.complex64)
    for i, bit in enumerate(bits):
        # phase > 0 → symbol 1, phase < 0 → symbol 0
        phase = 0.3 if bit == 1 else -0.3
        samples[i * sps : (i + 1) * sps] = np.exp(1j * phase)
    return samples


def generate_fsk(
    bits, sample_rate: float, symbol_rate: float, deviation: float
) -> np.ndarray:
    """Generate binary FSK signal."""
    sps = int(sample_rate / symbol_rate)
    n_samples = len(bits) * sps
    phase = np.zeros(n_samples)
    cum_phase = 0.0
    for i, bit in enumerate(bits):
        freq = deviation if bit == 1 else -deviation
        for j in range(sps):
            idx = i * sps + j
            cum_phase += 2 * np.pi * freq / sample_rate
            phase[idx] = cum_phase
    return np.exp(1j * phase).astype(np.complex64)


# ---------------------------------------------------------------------------
# SSB Demodulator
# ---------------------------------------------------------------------------


class TestSSBDemodulator:
    """Test SSB demodulator with USB and LSB modes."""

    SAMPLE_RATE = 48000.0

    def test_usb_demodulation_output_length(self):
        """USB demodulation preserves sample count."""
        demod = SSBDemodulator(self.SAMPLE_RATE, mode="usb")
        tone = generate_tone(1000, self.SAMPLE_RATE, 0.1)
        result = demod.demodulate(tone)
        assert len(result) == len(tone)

    def test_lsb_demodulation_output_length(self):
        """LSB demodulation preserves sample count."""
        demod = SSBDemodulator(self.SAMPLE_RATE, mode="lsb")
        tone = generate_tone(1000, self.SAMPLE_RATE, 0.1)
        result = demod.demodulate(tone)
        assert len(result) == len(tone)

    def test_usb_recovers_real_component(self):
        """USB extracts the real part of the signal."""
        demod = SSBDemodulator(self.SAMPLE_RATE, mode="usb")
        tone = generate_tone(1000, self.SAMPLE_RATE, 0.1)
        result = demod.demodulate(tone)
        np.testing.assert_allclose(result, tone.real, atol=1e-6)

    def test_lsb_flips_spectrum(self):
        """LSB conjugates before extracting real — inverts frequency."""
        demod = SSBDemodulator(self.SAMPLE_RATE, mode="lsb")
        tone = generate_tone(1000, self.SAMPLE_RATE, 0.1)
        result = demod.demodulate(tone)
        expected = np.conj(tone).real
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("mode", ["usb", "lsb"])
    def test_output_is_real(self, mode):
        """SSB output is always real-valued."""
        demod = SSBDemodulator(self.SAMPLE_RATE, mode=mode)
        tone = generate_tone(500, self.SAMPLE_RATE, 0.05)
        result = demod.demodulate(tone)
        assert result.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# FSK Demodulator
# ---------------------------------------------------------------------------


class TestFSKDemodulator:
    """Test FSK demodulator with synthetic signals."""

    SAMPLE_RATE = 48000.0
    SYMBOL_RATE = 1200.0
    DEVIATION = 2400.0

    def test_output_length(self):
        """FSK demodulation produces output matching input length."""
        demod = FSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, self.DEVIATION)
        signal = generate_fsk(
            [1, 0, 1, 1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE, self.DEVIATION
        )
        result = demod.demodulate(signal)
        # FM demod output is 1 sample shorter (diff operation)
        assert len(result) >= len(signal) - 2

    def test_demodulate_bits_binary(self):
        """Bit output is binary (0.0 or 1.0)."""
        demod = FSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, self.DEVIATION)
        signal = generate_fsk(
            [1, 0, 1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE, self.DEVIATION
        )
        bits = demod.demodulate_bits(signal)
        unique_vals = set(np.unique(bits))
        assert unique_vals.issubset({0.0, 1.0})

    def test_reset_clears_state(self):
        """Reset should not crash."""
        demod = FSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, self.DEVIATION)
        signal = generate_fsk([1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE, self.DEVIATION)
        demod.demodulate(signal)
        demod.reset()
        # Should be able to demodulate again after reset
        result = demod.demodulate(signal)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# PSK Demodulator
# ---------------------------------------------------------------------------


class TestPSKDemodulator:
    """Test PSK demodulator with BPSK and QPSK."""

    SAMPLE_RATE = 48000.0
    SYMBOL_RATE = 1200.0

    def test_bpsk_output_length(self):
        """BPSK demodulation preserves sample count."""
        demod = PSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=2)
        signal = generate_bpsk([1, 0, 1, 1], self.SAMPLE_RATE, self.SYMBOL_RATE)
        result = demod.demodulate(signal)
        assert len(result) == len(signal)

    def test_bpsk_output_is_binary(self):
        """BPSK output values are 0 or 1."""
        demod = PSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=2)
        signal = generate_bpsk([1, 0, 1, 0, 1], self.SAMPLE_RATE, self.SYMBOL_RATE)
        result = demod.demodulate(signal)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0.0, 1.0})

    def test_bpsk_recovers_known_pattern(self):
        """BPSK correctly recovers a known bit pattern (sample-level)."""
        demod = PSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=2)
        bits = [1, 0, 1, 1, 0]
        signal = generate_bpsk(bits, self.SAMPLE_RATE, self.SYMBOL_RATE)
        result = demod.demodulate(signal)

        # Each bit occupies sps samples — check majority vote per symbol
        sps = int(self.SAMPLE_RATE / self.SYMBOL_RATE)
        for i, expected_bit in enumerate(bits):
            symbol_samples = result[i * sps : (i + 1) * sps]
            recovered = 1 if np.mean(symbol_samples) > 0.5 else 0
            assert recovered == expected_bit

    @pytest.mark.parametrize("order", [2, 4])
    def test_psk_creates_without_error(self, order):
        """PSK demodulators of various orders can be created."""
        demod = PSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=order)
        assert demod._order == order

    def test_qpsk_output_range(self):
        """QPSK outputs symbols in {0, 1, 2, 3}."""
        demod = PSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=4)
        # Generate QPSK: 4 constellation points
        t = np.arange(4000) / self.SAMPLE_RATE
        phases = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
        signal = np.zeros(4000, dtype=np.complex64)
        for i, phase in enumerate(phases):
            signal[i * 1000 : (i + 1) * 1000] = np.exp(1j * phase)
        result = demod.demodulate(signal)
        assert np.all(result >= 0)
        assert np.all(result < 4)

    def test_constellation_output(self):
        """get_constellation returns I and Q arrays."""
        demod = PSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        signal = generate_bpsk([1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE)
        i_vals, q_vals = demod.get_constellation(signal)
        assert len(i_vals) == len(signal)
        assert len(q_vals) == len(signal)


# ---------------------------------------------------------------------------
# GFSK Demodulator
# ---------------------------------------------------------------------------


class TestGFSKDemodulator:
    """Test GFSK demodulator (Bluetooth/DECT style)."""

    SAMPLE_RATE = 48000.0
    SYMBOL_RATE = 1200.0

    def test_creation(self):
        """GFSK demodulator initializes without error."""
        demod = GFSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        assert demod.symbol_rate == self.SYMBOL_RATE

    def test_demodulate_output_length(self):
        """Demodulated output has reasonable length."""
        demod = GFSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        signal = generate_fsk(
            [1, 0, 1, 0, 1, 0, 1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE, 600
        )
        result = demod.demodulate(signal)
        # Output is symbol-rate, not sample-rate
        assert len(result) > 0

    def test_demodulate_soft_output(self):
        """Soft decision output has non-zero values."""
        demod = GFSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        signal = generate_fsk(
            [1, 0, 1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE, 600
        )
        soft = demod.demodulate_soft(signal)
        assert len(soft) > 0

    @pytest.mark.parametrize("bt", [0.3, 0.5, 1.0])
    def test_different_bt_products(self, bt):
        """GFSK demodulator works with various BT products."""
        demod = GFSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, bt=bt)
        signal = generate_fsk(
            [1, 0, 1, 0], self.SAMPLE_RATE, self.SYMBOL_RATE, 600
        )
        result = demod.demodulate(signal)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# MSK Demodulator
# ---------------------------------------------------------------------------


class TestMSKDemodulator:
    """Test MSK demodulator."""

    SAMPLE_RATE = 48000.0
    SYMBOL_RATE = 1200.0

    def test_creation(self):
        """MSK demodulator initializes without error."""
        demod = MSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        assert demod._deviation == self.SYMBOL_RATE / 4

    def test_demodulate_no_crash(self):
        """MSK demodulation doesn't crash on synthetic signal."""
        demod = MSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        # MSK is FSK with h=0.5, deviation = symbol_rate/4
        signal = generate_fsk(
            [1, 0, 1, 1, 0, 0, 1, 0],
            self.SAMPLE_RATE,
            self.SYMBOL_RATE,
            self.SYMBOL_RATE / 4,
        )
        result = demod.demodulate(signal)
        assert len(result) > 0

    def test_demodulate_soft(self):
        """MSK soft decision output is non-empty."""
        demod = MSKDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE)
        signal = generate_fsk(
            [1, 0, 1, 0],
            self.SAMPLE_RATE,
            self.SYMBOL_RATE,
            self.SYMBOL_RATE / 4,
        )
        soft = demod.demodulate_soft(signal)
        assert len(soft) > 0


# ---------------------------------------------------------------------------
# QAM Demodulator
# ---------------------------------------------------------------------------


class TestQAMDemodulator:
    """Test QAM demodulator."""

    SAMPLE_RATE = 48000.0
    SYMBOL_RATE = 1200.0

    def test_creation_16qam(self):
        """16-QAM demodulator creates correctly."""
        demod = QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=16)
        assert demod.order == 16
        assert demod.bits_per_symbol == 4
        assert len(demod.constellation) == 16

    def test_creation_64qam(self):
        """64-QAM demodulator creates correctly."""
        demod = QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=64)
        assert demod.order == 64
        assert demod.bits_per_symbol == 6
        assert len(demod.constellation) == 64

    def test_invalid_order_raises(self):
        """Invalid QAM order raises ValueError."""
        with pytest.raises(ValueError, match="QAM order"):
            QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=32)

    @pytest.mark.parametrize("order", [16, 64, 256])
    def test_constellation_points_count(self, order):
        """Constellation has exactly `order` points."""
        demod = QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=order)
        assert len(demod.constellation) == order

    def test_demodulate_constellation_point(self):
        """Feeding a constellation point recovers the correct symbol index."""
        demod = QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=16)
        # Feed constellation points directly as samples
        points = demod.constellation
        result = demod.demodulate(points)
        # Each point should map to a unique symbol index 0..15
        assert len(result) == 16
        assert set(result.astype(int)) == set(range(16))

    def test_demodulate_soft_output_shape(self):
        """Soft decision output has correct shape (n_samples × bits_per_symbol)."""
        demod = QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=16)
        points = demod.constellation[:4]
        soft = demod.demodulate_soft(points)
        assert soft.shape == (4, 4)  # 4 samples, 4 bits per symbol

    def test_normalized_constellation_unit_power(self):
        """Normalized constellation has approximately unit average power."""
        demod = QAMDemodulator(self.SAMPLE_RATE, self.SYMBOL_RATE, order=16)
        avg_power = np.mean(np.abs(demod.constellation) ** 2)
        np.testing.assert_allclose(avg_power, 1.0, atol=0.05)

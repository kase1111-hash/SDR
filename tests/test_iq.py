"""Tests for I/Q sample utilities."""

import os
import tempfile

import numpy as np
import pytest

from sdr_module.utils.iq import (
    apply_dc_offset_correction,
    apply_iq_imbalance_correction,
    complex_to_interleaved,
    complex_to_iq,
    estimate_iq_imbalance,
    interleaved_to_complex,
    iq_to_complex,
    load_iq_file,
    save_iq_file,
)


class TestIQToComplex:
    """Tests for iq_to_complex function."""

    def test_basic_conversion(self):
        """Test basic I/Q to complex conversion."""
        i = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        result = iq_to_complex(i, q)

        expected = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer_input(self):
        """Test conversion with integer input."""
        i = np.array([1, 2, 3])
        q = np.array([4, 5, 6])

        result = iq_to_complex(i, q)

        assert result.dtype == np.complex64
        assert result[0] == 1 + 4j

    def test_output_dtype(self):
        """Test output is complex64."""
        i = np.array([1.0], dtype=np.float64)
        q = np.array([1.0], dtype=np.float64)

        result = iq_to_complex(i, q)

        # Real and imag parts should be float32
        assert result.real.dtype == np.float32


class TestComplexToIQ:
    """Tests for complex_to_iq function."""

    def test_basic_conversion(self):
        """Test basic complex to I/Q conversion."""
        samples = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)

        i, q = complex_to_iq(samples)

        np.testing.assert_array_almost_equal(i, [1, 3, 5])
        np.testing.assert_array_almost_equal(q, [2, 4, 6])

    def test_output_dtype(self):
        """Test output is float32."""
        samples = np.array([1 + 2j], dtype=np.complex128)

        i, q = complex_to_iq(samples)

        assert i.dtype == np.float32
        assert q.dtype == np.float32

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original_i = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_q = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        complex_samples = iq_to_complex(original_i, original_q)
        i, q = complex_to_iq(complex_samples)

        np.testing.assert_array_almost_equal(i, original_i)
        np.testing.assert_array_almost_equal(q, original_q)


class TestInterleavedToComplex:
    """Tests for interleaved_to_complex function."""

    def test_uint8_conversion(self):
        """Test uint8 interleaved conversion."""
        # 127 should map to ~0, 255 to ~1, 0 to ~-1
        data = np.array([127, 127, 255, 127, 0, 127], dtype=np.uint8)

        result = interleaved_to_complex(data, np.uint8)

        assert len(result) == 3
        assert abs(result[0].real) < 0.1  # ~0
        assert result[1].real > 0.9  # ~1

    def test_int8_conversion(self):
        """Test int8 interleaved conversion."""
        data = np.array([0, 0, 127, 0, -127, 0], dtype=np.int8)

        result = interleaved_to_complex(data, np.int8)

        assert len(result) == 3
        assert abs(result[0].real) < 0.1
        assert result[1].real > 0.9

    def test_int16_conversion(self):
        """Test int16 interleaved conversion."""
        data = np.array([0, 0, 32767, 0], dtype=np.int16)

        result = interleaved_to_complex(data, np.int16)

        assert len(result) == 2
        assert result[1].real > 0.9

    def test_float32_passthrough(self):
        """Test float32 passes through unchanged."""
        data = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        result = interleaved_to_complex(data, np.float32)

        assert len(result) == 2
        assert abs(result[0].real - 0.5) < 0.01


class TestComplexToInterleaved:
    """Tests for complex_to_interleaved function."""

    def test_uint8_output(self):
        """Test uint8 interleaved output."""
        samples = np.array([0 + 0j, 1 + 0j, -1 + 0j], dtype=np.complex64)

        result = complex_to_interleaved(samples, np.uint8)

        assert len(result) == 6
        assert result.dtype == np.uint8
        # 0 maps to 127.5, 1 maps to 255, -1 maps to 0
        assert abs(result[0] - 127) < 2  # ~127
        assert result[2] == 255  # 1 -> 255
        assert result[4] == 0  # -1 -> 0

    def test_int8_output(self):
        """Test int8 interleaved output."""
        samples = np.array([1 + 0j, -1 + 0j], dtype=np.complex64)

        result = complex_to_interleaved(samples, np.int8)

        assert result.dtype == np.int8
        assert result[0] == 127  # 1 -> 127
        assert result[2] == -127  # -1 -> -127

    def test_int16_output(self):
        """Test int16 interleaved output."""
        samples = np.array([1 + 0j], dtype=np.complex64)

        result = complex_to_interleaved(samples, np.int16)

        assert result.dtype == np.int16
        assert result[0] == 32767

    def test_float32_output(self):
        """Test float32 interleaved output."""
        samples = np.array([0.5 + 0.25j], dtype=np.complex64)

        result = complex_to_interleaved(samples, np.float32)

        assert result.dtype == np.float32
        assert abs(result[0] - 0.5) < 0.01
        assert abs(result[1] - 0.25) < 0.01

    def test_clipping(self):
        """Test values are clipped to valid range."""
        samples = np.array([2 + 3j], dtype=np.complex64)  # Out of range

        result = complex_to_interleaved(samples, np.float32)

        assert result[0] == 1.0  # Clipped
        assert result[1] == 1.0  # Clipped

    def test_roundtrip_uint8(self):
        """Test uint8 roundtrip conversion."""
        original = np.array([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)

        interleaved = complex_to_interleaved(original, np.uint8)
        recovered = interleaved_to_complex(interleaved, np.uint8)

        # Allow for quantization error
        np.testing.assert_array_almost_equal(original, recovered, decimal=1)


class TestIQFileOperations:
    """Tests for I/Q file load/save operations."""

    def test_save_and_load_cf32(self):
        """Test saving and loading cf32 format."""
        samples = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)

        with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
            filepath = f.name

        try:
            save_iq_file(samples, filepath, "cf32")
            loaded = load_iq_file(filepath, "cf32")

            np.testing.assert_array_almost_equal(samples, loaded)
        finally:
            os.unlink(filepath)

    def test_save_and_load_cu8(self):
        """Test saving and loading cu8 format."""
        samples = np.array([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)

        with tempfile.NamedTemporaryFile(suffix=".cu8", delete=False) as f:
            filepath = f.name

        try:
            save_iq_file(samples, filepath, "cu8")
            loaded = load_iq_file(filepath, "cu8")

            # Allow for quantization error
            np.testing.assert_array_almost_equal(samples, loaded, decimal=1)
        finally:
            os.unlink(filepath)

    def test_load_with_offset(self):
        """Test loading with sample offset."""
        samples = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64)

        with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
            filepath = f.name

        try:
            save_iq_file(samples, filepath, "cf32")
            loaded = load_iq_file(filepath, "cf32", offset_samples=2)

            np.testing.assert_array_almost_equal(samples[2:], loaded)
        finally:
            os.unlink(filepath)

    def test_load_with_count(self):
        """Test loading limited number of samples."""
        samples = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64)

        with tempfile.NamedTemporaryFile(suffix=".cf32", delete=False) as f:
            filepath = f.name

        try:
            save_iq_file(samples, filepath, "cf32")
            loaded = load_iq_file(filepath, "cf32", num_samples=2)

            np.testing.assert_array_almost_equal(samples[:2], loaded)
        finally:
            os.unlink(filepath)

    def test_unsupported_load_format(self):
        """Test loading unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                load_iq_file(filepath, "xyz")
        finally:
            os.unlink(filepath)

    def test_unsupported_save_format(self):
        """Test saving unsupported format raises error."""
        samples = np.array([1 + 1j], dtype=np.complex64)

        with pytest.raises(ValueError, match="Unsupported format"):
            save_iq_file(samples, "/tmp/test.xyz", "xyz")


class TestDCOffsetCorrection:
    """Tests for DC offset correction."""

    def test_removes_dc_offset(self):
        """Test DC offset is removed."""
        # Create signal with DC offset
        offset = 0.5 + 0.3j
        samples = np.random.randn(1000).astype(np.complex64) + offset

        corrected = apply_dc_offset_correction(samples)

        assert abs(np.mean(corrected)) < 0.1

    def test_zero_mean_unchanged(self):
        """Test zero-mean signal is mostly unchanged."""
        samples = np.random.randn(1000).astype(np.complex64)
        samples = samples - np.mean(samples)  # Ensure zero mean

        corrected = apply_dc_offset_correction(samples)

        # Should be essentially the same
        np.testing.assert_array_almost_equal(samples, corrected, decimal=5)


class TestIQImbalanceCorrection:
    """Tests for I/Q imbalance correction."""

    def test_no_imbalance(self):
        """Test no correction needed for balanced signal."""
        samples = np.exp(1j * np.linspace(0, 10 * np.pi, 1000)).astype(np.complex64)

        corrected = apply_iq_imbalance_correction(samples, 0.0, 0.0)

        np.testing.assert_array_almost_equal(samples, corrected)

    def test_gain_imbalance_correction(self):
        """Test gain imbalance is corrected."""
        # Create signal with gain imbalance
        t = np.linspace(0, 10 * np.pi, 1000)
        i = np.cos(t)
        q = 0.8 * np.sin(t)  # Q has 20% less gain
        samples = (i + 1j * q).astype(np.complex64)

        corrected = apply_iq_imbalance_correction(samples, gain_imbalance=-0.2)

        # Q should now have similar power to I
        i_power = np.mean(corrected.real**2)
        q_power = np.mean(corrected.imag**2)
        assert abs(i_power - q_power) / i_power < 0.1

    def test_phase_imbalance_correction(self):
        """Test phase imbalance correction."""
        samples = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex64)

        # Apply small phase correction
        corrected = apply_iq_imbalance_correction(samples, phase_imbalance=0.1)

        assert isinstance(corrected, np.ndarray)
        assert len(corrected) == 4


class TestEstimateIQImbalance:
    """Tests for I/Q imbalance estimation."""

    def test_balanced_signal(self):
        """Test balanced signal shows no imbalance."""
        t = np.linspace(0, 100 * np.pi, 10000)
        samples = np.exp(1j * t).astype(np.complex64)

        gain, phase = estimate_iq_imbalance(samples)

        assert abs(gain) < 0.1
        assert abs(phase) < 0.1

    def test_gain_imbalance_detection(self):
        """Test gain imbalance is detected."""
        t = np.linspace(0, 100 * np.pi, 10000)
        i = np.cos(t)
        q = 0.7 * np.sin(t)  # 30% gain reduction
        samples = (i + 1j * q).astype(np.complex64)

        gain, phase = estimate_iq_imbalance(samples)

        # Should detect ~0.43 gain imbalance (sqrt(1/0.49) - 1)
        assert gain > 0.3

    def test_returns_floats(self):
        """Test returns float values."""
        samples = np.random.randn(100).astype(np.complex64) + 1j * np.random.randn(
            100
        ).astype(np.complex64)

        gain, phase = estimate_iq_imbalance(samples)

        assert isinstance(gain, (float, np.floating))
        assert isinstance(phase, (float, np.floating))

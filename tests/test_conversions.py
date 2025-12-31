"""Tests for conversion utilities."""

import numpy as np
import pytest

from sdr_module.utils.conversions import (
    bandwidth_to_str,
    db_to_linear,
    dbm_to_dbv,
    dbm_to_watts,
    freq_to_str,
    linear_to_db,
    sample_rate_to_str,
    str_to_freq,
    watts_to_dbm,
)


class TestPowerConversions:
    """Test power conversion functions."""

    def test_db_to_linear(self):
        """Test dB to linear conversion."""
        assert db_to_linear(0) == pytest.approx(1.0)
        assert db_to_linear(10) == pytest.approx(10.0)
        assert db_to_linear(20) == pytest.approx(100.0)
        assert db_to_linear(-10) == pytest.approx(0.1)

    def test_linear_to_db(self):
        """Test linear to dB conversion."""
        assert linear_to_db(1.0) == pytest.approx(0.0)
        assert linear_to_db(10.0) == pytest.approx(10.0)
        assert linear_to_db(100.0) == pytest.approx(20.0)
        assert linear_to_db(0.1) == pytest.approx(-10.0)

    def test_db_linear_roundtrip(self):
        """Test roundtrip conversion dB <-> linear."""
        values = [0.001, 0.1, 1, 10, 100]
        for val in values:
            assert linear_to_db(db_to_linear(linear_to_db(val))) == pytest.approx(
                linear_to_db(val)
            )

    def test_dbm_to_watts(self):
        """Test dBm to watts conversion."""
        assert dbm_to_watts(0) == pytest.approx(0.001)  # 0 dBm = 1 mW
        assert dbm_to_watts(30) == pytest.approx(1.0)  # 30 dBm = 1 W
        assert dbm_to_watts(-30) == pytest.approx(1e-6)  # -30 dBm = 1 µW

    def test_watts_to_dbm(self):
        """Test watts to dBm conversion."""
        assert watts_to_dbm(0.001) == pytest.approx(0.0)  # 1 mW = 0 dBm
        assert watts_to_dbm(1.0) == pytest.approx(30.0)  # 1 W = 30 dBm
        assert watts_to_dbm(1e-6) == pytest.approx(-30.0)  # 1 µW = -30 dBm

    def test_dbm_watts_roundtrip(self):
        """Test roundtrip conversion dBm <-> watts."""
        values = [1e-9, 1e-6, 0.001, 1, 100]
        for val in values:
            assert dbm_to_watts(watts_to_dbm(val)) == pytest.approx(val)

    def test_dbm_to_dbv(self):
        """Test dBm to dBV conversion."""
        # At 50 ohms, 0 dBm should give approximately -13 dBV
        result = dbm_to_dbv(0, impedance=50.0)
        assert result == pytest.approx(-13.0, abs=0.1)

    def test_array_conversions(self):
        """Test conversions work with numpy arrays."""
        arr = np.array([0, 10, 20, -10])
        linear = db_to_linear(arr)
        assert isinstance(linear, np.ndarray)
        assert len(linear) == 4
        assert linear[0] == pytest.approx(1.0)
        assert linear[1] == pytest.approx(10.0)


class TestFrequencyConversions:
    """Test frequency conversion and formatting functions."""

    def test_freq_to_str_hz(self):
        """Test Hz formatting."""
        assert "Hz" in freq_to_str(100)
        assert "100" in freq_to_str(100)

    def test_freq_to_str_khz(self):
        """Test kHz formatting."""
        result = freq_to_str(7100)
        assert "kHz" in result
        assert "7.1" in result

    def test_freq_to_str_mhz(self):
        """Test MHz formatting."""
        result = freq_to_str(144.2e6)
        assert "MHz" in result
        assert "144.2" in result

    def test_freq_to_str_ghz(self):
        """Test GHz formatting."""
        result = freq_to_str(2.4e9)
        assert "GHz" in result
        assert "2.4" in result

    def test_str_to_freq_hz(self):
        """Test parsing Hz strings."""
        assert str_to_freq("100Hz") == 100.0
        assert str_to_freq("100 Hz") == 100.0
        assert str_to_freq("100") == 100.0

    def test_str_to_freq_khz(self):
        """Test parsing kHz strings."""
        assert str_to_freq("7.1kHz") == 7100.0
        assert str_to_freq("7.1 kHz") == 7100.0
        assert str_to_freq("7100kHz") == 7.1e6

    def test_str_to_freq_mhz(self):
        """Test parsing MHz strings."""
        assert str_to_freq("144.2MHz") == 144.2e6
        assert str_to_freq("144.2 MHz") == 144.2e6
        assert str_to_freq("100M") == 100e6

    def test_str_to_freq_ghz(self):
        """Test parsing GHz strings."""
        assert str_to_freq("2.4GHz") == 2.4e9
        assert str_to_freq("2.4 GHz") == 2.4e9
        assert str_to_freq("5G") == 5e9

    def test_freq_roundtrip(self):
        """Test roundtrip frequency conversions."""
        freqs = [100, 7.1e3, 144.2e6, 2.4e9]
        for freq in freqs:
            freq_str = freq_to_str(freq)
            parsed = str_to_freq(freq_str)
            assert parsed == pytest.approx(freq, rel=1e-5)

    def test_sample_rate_to_str(self):
        """Test sample rate formatting."""
        assert "S/s" in sample_rate_to_str(1000)
        assert "kS/s" in sample_rate_to_str(1e3)
        assert "MS/s" in sample_rate_to_str(2.4e6)

    def test_bandwidth_to_str(self):
        """Test bandwidth formatting."""
        result = bandwidth_to_str(20e6)
        assert "BW" in result
        assert "MHz" in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_handling(self):
        """Test handling of zero values."""
        # Should not raise exceptions
        linear_to_db(0)
        watts_to_dbm(0)

    def test_negative_power_invalid(self):
        """Test that negative linear power is handled."""
        # Should not crash, though result may not be meaningful
        result = linear_to_db(-1)
        assert isinstance(result, (float, np.ndarray))

    def test_case_insensitive_freq_parsing(self):
        """Test case-insensitive frequency parsing."""
        assert str_to_freq("100mhz") == str_to_freq("100MHz")
        assert str_to_freq("2.4ghz") == str_to_freq("2.4GHz")

    def test_whitespace_handling(self):
        """Test whitespace handling in frequency strings."""
        assert str_to_freq("  100  MHz  ") == 100e6
        assert str_to_freq("144.2MHz") == str_to_freq("144.2 MHz")

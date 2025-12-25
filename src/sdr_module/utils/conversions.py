"""
Unit conversion utilities for SDR applications.
"""

import numpy as np
from typing import Union

# Type alias for numeric types
Numeric = Union[float, int, np.ndarray]


def db_to_linear(db: Numeric) -> Numeric:
    """
    Convert decibels to linear (power ratio).

    Args:
        db: Value in dB

    Returns:
        Linear value
    """
    return 10 ** (db / 10)


def linear_to_db(linear: Numeric) -> Numeric:
    """
    Convert linear (power ratio) to decibels.

    Args:
        linear: Linear value

    Returns:
        Value in dB
    """
    return 10 * np.log10(linear + 1e-20)


def dbm_to_watts(dbm: Numeric) -> Numeric:
    """
    Convert dBm to Watts.

    Args:
        dbm: Power in dBm

    Returns:
        Power in Watts
    """
    return 10 ** ((dbm - 30) / 10)


def watts_to_dbm(watts: Numeric) -> Numeric:
    """
    Convert Watts to dBm.

    Args:
        watts: Power in Watts

    Returns:
        Power in dBm
    """
    return 10 * np.log10(watts + 1e-20) + 30


def dbm_to_dbv(dbm: Numeric, impedance: float = 50.0) -> Numeric:
    """
    Convert dBm to dBV.

    Args:
        dbm: Power in dBm
        impedance: Load impedance in ohms

    Returns:
        Voltage in dBV
    """
    # P = V^2/R, so V = sqrt(P*R)
    # dBV = dBm - 10*log10(1000) + 10*log10(R)
    return dbm - 30 + 10 * np.log10(impedance)


def freq_to_str(freq_hz: float) -> str:
    """
    Convert frequency to human-readable string.

    Args:
        freq_hz: Frequency in Hz

    Returns:
        Formatted string (e.g., "144.200 MHz")
    """
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.6f} GHz"
    elif freq_hz >= 1e6:
        return f"{freq_hz / 1e6:.6f} MHz"
    elif freq_hz >= 1e3:
        return f"{freq_hz / 1e3:.3f} kHz"
    else:
        return f"{freq_hz:.1f} Hz"


def str_to_freq(freq_str: str) -> float:
    """
    Parse frequency string to Hz.

    Args:
        freq_str: Frequency string (e.g., "144.2MHz", "7.1 MHz")

    Returns:
        Frequency in Hz
    """
    freq_str = freq_str.strip().upper()

    multipliers = {
        "GHZ": 1e9,
        "MHZ": 1e6,
        "KHZ": 1e3,
        "HZ": 1,
        "G": 1e9,
        "M": 1e6,
        "K": 1e3,
    }

    for suffix, mult in multipliers.items():
        if freq_str.endswith(suffix):
            value = freq_str[:-len(suffix)].strip()
            return float(value) * mult

    # No suffix, assume Hz
    return float(freq_str)


def sample_rate_to_str(rate_hz: float) -> str:
    """
    Convert sample rate to human-readable string.

    Args:
        rate_hz: Sample rate in Hz

    Returns:
        Formatted string (e.g., "2.4 MS/s")
    """
    if rate_hz >= 1e6:
        return f"{rate_hz / 1e6:.2f} MS/s"
    elif rate_hz >= 1e3:
        return f"{rate_hz / 1e3:.2f} kS/s"
    else:
        return f"{rate_hz:.0f} S/s"


def bandwidth_to_str(bw_hz: float) -> str:
    """
    Convert bandwidth to human-readable string.

    Args:
        bw_hz: Bandwidth in Hz

    Returns:
        Formatted string
    """
    return freq_to_str(bw_hz).replace("Hz", "Hz BW")

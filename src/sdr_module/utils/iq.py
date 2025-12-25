"""
I/Q sample utilities for SDR applications.

Provides conversion between various I/Q data formats.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def iq_to_complex(i_samples: np.ndarray, q_samples: np.ndarray) -> np.ndarray:
    """
    Convert separate I and Q arrays to complex.

    Args:
        i_samples: In-phase samples
        q_samples: Quadrature samples

    Returns:
        Complex numpy array
    """
    return i_samples.astype(np.float32) + 1j * q_samples.astype(np.float32)


def complex_to_iq(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert complex samples to separate I and Q arrays.

    Args:
        samples: Complex samples

    Returns:
        Tuple of (I, Q) arrays
    """
    return samples.real.astype(np.float32), samples.imag.astype(np.float32)


def interleaved_to_complex(
    data: np.ndarray,
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Convert interleaved I/Q data to complex.

    Args:
        data: Interleaved [I0, Q0, I1, Q1, ...] data
        dtype: Original data type (uint8, int8, int16, float32)

    Returns:
        Complex numpy array normalized to [-1, 1]
    """
    data = data.astype(np.float32)

    # Normalize based on original dtype
    if dtype == np.uint8:
        data = (data - 127.5) / 127.5
    elif dtype == np.int8:
        data = data / 127.0
    elif dtype == np.int16:
        data = data / 32767.0
    # float32 is already normalized

    # Reshape and convert to complex
    iq = data.reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_interleaved(
    samples: np.ndarray,
    dtype: np.dtype = np.uint8
) -> np.ndarray:
    """
    Convert complex samples to interleaved I/Q data.

    Args:
        samples: Complex samples (assumed normalized to [-1, 1])
        dtype: Target data type

    Returns:
        Interleaved [I0, Q0, I1, Q1, ...] data
    """
    # Clip to valid range
    i_data = np.clip(samples.real, -1, 1)
    q_data = np.clip(samples.imag, -1, 1)

    # Scale based on target dtype
    if dtype == np.uint8:
        i_data = (i_data * 127.5 + 127.5).astype(np.uint8)
        q_data = (q_data * 127.5 + 127.5).astype(np.uint8)
    elif dtype == np.int8:
        i_data = (i_data * 127).astype(np.int8)
        q_data = (q_data * 127).astype(np.int8)
    elif dtype == np.int16:
        i_data = (i_data * 32767).astype(np.int16)
        q_data = (q_data * 32767).astype(np.int16)
    else:  # float32
        i_data = i_data.astype(np.float32)
        q_data = q_data.astype(np.float32)

    # Interleave
    result = np.empty(len(samples) * 2, dtype=dtype)
    result[0::2] = i_data
    result[1::2] = q_data

    return result


def load_iq_file(
    filepath: str,
    format: str = "cu8",
    num_samples: Optional[int] = None,
    offset_samples: int = 0
) -> np.ndarray:
    """
    Load I/Q samples from file.

    Args:
        filepath: Path to I/Q file
        format: File format (cu8, cs8, cs16, cf32)
        num_samples: Number of samples to read (None = all)
        offset_samples: Number of samples to skip

    Returns:
        Complex numpy array
    """
    format = format.lower()

    # Determine dtype and bytes per sample
    format_info = {
        "cu8": (np.uint8, 2),
        "cs8": (np.int8, 2),
        "cs16": (np.int16, 4),
        "cf32": (np.float32, 8),
        "cf64": (np.float64, 16),
    }

    if format not in format_info:
        raise ValueError(f"Unsupported format: {format}")

    dtype, bytes_per_sample = format_info[format]

    # Read raw data
    offset_bytes = offset_samples * bytes_per_sample

    with open(filepath, 'rb') as f:
        f.seek(offset_bytes)
        if num_samples is not None:
            data = np.fromfile(f, dtype=dtype, count=num_samples * 2)
        else:
            data = np.fromfile(f, dtype=dtype)

    # Convert to complex
    if format in ("cf32", "cf64"):
        # Already complex-formatted
        return data.view(np.complex64 if format == "cf32" else np.complex128)
    else:
        return interleaved_to_complex(data, dtype)


def save_iq_file(
    samples: np.ndarray,
    filepath: str,
    format: str = "cf32"
) -> None:
    """
    Save I/Q samples to file.

    Args:
        samples: Complex samples
        filepath: Output file path
        format: File format (cu8, cs8, cs16, cf32)
    """
    format = format.lower()

    format_dtypes = {
        "cu8": np.uint8,
        "cs8": np.int8,
        "cs16": np.int16,
        "cf32": np.float32,
    }

    if format not in format_dtypes:
        raise ValueError(f"Unsupported format: {format}")

    dtype = format_dtypes[format]

    if format == "cf32":
        # Direct complex output
        data = samples.astype(np.complex64)
    else:
        data = complex_to_interleaved(samples, dtype)

    data.tofile(filepath)


def apply_dc_offset_correction(samples: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from I/Q samples.

    Args:
        samples: Complex samples

    Returns:
        DC-corrected samples
    """
    return samples - np.mean(samples)


def apply_iq_imbalance_correction(
    samples: np.ndarray,
    gain_imbalance: float = 0.0,
    phase_imbalance: float = 0.0
) -> np.ndarray:
    """
    Correct I/Q gain and phase imbalance.

    Args:
        samples: Complex samples
        gain_imbalance: Gain imbalance factor
        phase_imbalance: Phase imbalance in radians

    Returns:
        Corrected samples
    """
    # Apply corrections
    i = samples.real
    q = samples.imag

    # Gain correction
    q_corrected = q / (1 + gain_imbalance)

    # Phase correction
    i_corrected = i - q_corrected * np.sin(phase_imbalance)
    q_corrected = q_corrected * np.cos(phase_imbalance)

    return i_corrected + 1j * q_corrected


def estimate_iq_imbalance(samples: np.ndarray) -> Tuple[float, float]:
    """
    Estimate I/Q gain and phase imbalance.

    Args:
        samples: Complex samples

    Returns:
        Tuple of (gain_imbalance, phase_imbalance_radians)
    """
    i = samples.real
    q = samples.imag

    # Estimate gain imbalance
    i_power = np.mean(i**2)
    q_power = np.mean(q**2)
    gain_imbalance = np.sqrt(i_power / (q_power + 1e-10)) - 1

    # Estimate phase imbalance
    cross_corr = np.mean(i * q)
    phase_imbalance = np.arcsin(cross_corr / (np.sqrt(i_power * q_power) + 1e-10))

    return gain_imbalance, phase_imbalance

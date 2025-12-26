"""
Noise Filter Plugin.

Provides adaptive noise reduction for SDR signals using
spectral subtraction and Wiener filtering techniques.
"""

import numpy as np
from typing import Dict, Any, Optional

from sdr_module.plugins import (
    ProcessorPlugin,
    PluginMetadata,
    PluginType,
)


class NoiseFilterPlugin(ProcessorPlugin):
    """
    Adaptive noise filter using spectral subtraction.

    Estimates noise floor during quiet periods and subtracts
    it from the signal to improve SNR.
    """

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="noise_filter",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
            author="SDR Module Team",
            description="Adaptive noise reduction using spectral subtraction",
            tags=["processor", "filter", "noise", "dsp"],
            license="MIT",
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the noise filter."""
        self._sample_rate = config.get("sample_rate", 2.4e6)
        self._fft_size = config.get("fft_size", 1024)
        self._noise_floor: Optional[np.ndarray] = None
        self._alpha = config.get("alpha", 0.98)  # Noise estimate smoothing
        self._beta = config.get("beta", 0.02)    # Subtraction factor
        self._noise_update = config.get("noise_update", True)

        # Spectral floor estimate (in dB)
        self._spectral_floor = np.zeros(self._fft_size)
        self._frame_count = 0

        return True

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "name": "Noise Filter",
            "category": "filter",
            "input_type": "complex",
            "output_type": "complex",
            "description": "Adaptive spectral noise reduction",
            "parameters": {
                "alpha": "Noise estimate smoothing (0-1)",
                "beta": "Subtraction strength (0-1)",
                "fft_size": "FFT size for spectral analysis",
                "noise_update": "Enable automatic noise floor updates",
            },
        }

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to samples.

        Uses overlap-add spectral subtraction for noise reduction
        while preserving signal phase.

        Args:
            samples: Complex I/Q samples

        Returns:
            Noise-reduced samples
        """
        if len(samples) < self._fft_size:
            return samples

        # Process in overlapping frames
        hop_size = self._fft_size // 2
        num_frames = (len(samples) - self._fft_size) // hop_size + 1

        # Output buffer
        output = np.zeros(len(samples), dtype=np.complex128)
        window = np.hanning(self._fft_size)

        for i in range(num_frames):
            start = i * hop_size
            end = start + self._fft_size

            frame = samples[start:end] * window

            # FFT
            spectrum = np.fft.fft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Update noise estimate
            if self._noise_update:
                self._update_noise_estimate(magnitude)

            # Apply spectral subtraction
            if self._noise_floor is not None:
                # Subtract noise floor
                clean_magnitude = np.maximum(
                    magnitude - self._beta * self._noise_floor,
                    0.1 * magnitude  # Floor to prevent musical noise
                )
            else:
                clean_magnitude = magnitude

            # Reconstruct with original phase
            clean_spectrum = clean_magnitude * np.exp(1j * phase)

            # IFFT
            clean_frame = np.fft.ifft(clean_spectrum)

            # Overlap-add
            output[start:end] += clean_frame * window

            self._frame_count += 1

        # Normalize overlap regions
        output /= 1.5  # Hanning window overlap factor

        return output[:len(samples)]

    def _update_noise_estimate(self, magnitude: np.ndarray) -> None:
        """Update noise floor estimate using minimum statistics."""
        if self._noise_floor is None:
            self._noise_floor = magnitude.copy()
        else:
            # Exponential smoothing toward minimum
            self._noise_floor = np.minimum(
                self._noise_floor,
                self._alpha * self._noise_floor + (1 - self._alpha) * magnitude
            )

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a filter parameter."""
        if name == "alpha":
            if 0 <= value <= 1:
                self._alpha = float(value)
                return True
        elif name == "beta":
            if 0 <= value <= 1:
                self._beta = float(value)
                return True
        elif name == "fft_size":
            if value in (256, 512, 1024, 2048, 4096):
                self._fft_size = int(value)
                self.reset()
                return True
        elif name == "noise_update":
            self._noise_update = bool(value)
            return True

        return False

    def get_parameter(self, name: str) -> Optional[Any]:
        """Get a filter parameter."""
        params = {
            "alpha": self._alpha,
            "beta": self._beta,
            "fft_size": self._fft_size,
            "noise_update": self._noise_update,
            "frame_count": self._frame_count,
        }
        return params.get(name)

    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameters."""
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "fft_size": self._fft_size,
            "noise_update": self._noise_update,
        }

    def reset(self) -> None:
        """Reset filter state."""
        self._noise_floor = None
        self._spectral_floor = np.zeros(self._fft_size)
        self._frame_count = 0

    def get_noise_floor(self) -> Optional[np.ndarray]:
        """Get current noise floor estimate."""
        return self._noise_floor.copy() if self._noise_floor is not None else None

    def set_noise_floor(self, floor: np.ndarray) -> None:
        """Set noise floor manually (e.g., from calibration)."""
        if len(floor) == self._fft_size:
            self._noise_floor = floor.copy()
            self._noise_update = False  # Disable auto-update when manually set

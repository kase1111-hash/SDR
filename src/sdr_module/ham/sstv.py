"""
SSTV (Slow Scan Television) decoder for receiving images from space.

Supports decoding SSTV transmissions from:
- International Space Station (ISS) on 145.800 MHz
- Amateur radio operators
- Weather satellites

Common SSTV modes:
- PD120, PD180 (ISS favorites)
- Martin M1, M2
- Scottie S1, S2
- Robot 36, 72

Usage:
    decoder = SSTVDecoder(sample_rate=48000)
    decoder.process_audio(audio_samples)

    if decoder.is_complete():
        image = decoder.get_image()
        decoder.save_image("iss_capture.png")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SSTVMode(Enum):
    """SSTV transmission modes."""

    # Robot modes
    ROBOT_36 = auto()  # 36 seconds, 320x240
    ROBOT_72 = auto()  # 72 seconds, 320x240

    # Scottie modes
    SCOTTIE_1 = auto()  # 110 seconds, 320x256
    SCOTTIE_2 = auto()  # 71 seconds, 320x256
    SCOTTIE_DX = auto()  # 269 seconds, 320x256

    # Martin modes
    MARTIN_1 = auto()  # 114 seconds, 320x256
    MARTIN_2 = auto()  # 58 seconds, 320x256

    # PD modes (ISS favorites)
    PD_90 = auto()  # 90 seconds, 320x256
    PD_120 = auto()  # 120 seconds, 640x496
    PD_160 = auto()  # 160 seconds, 512x400
    PD_180 = auto()  # 180 seconds, 640x496
    PD_240 = auto()  # 240 seconds, 640x496
    PD_290 = auto()  # 290 seconds, 800x616

    # Wraase modes
    WRAASE_SC2_180 = auto()

    UNKNOWN = auto()


@dataclass
class SSTVModeSpec:
    """Specification for an SSTV mode."""

    name: str
    mode: SSTVMode
    vis_code: int  # Vertical Interval Signaling code
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    scan_time_ms: float  # Time for one line in milliseconds
    sync_pulse_ms: float  # Sync pulse duration
    sync_porch_ms: float  # Sync porch duration
    color_format: str  # "RGB", "YUV", "GBR"
    channel_times_ms: List[float]  # Time for each color channel


# SSTV Mode specifications
SSTV_MODES = {
    # Robot modes - YUV format
    0x08: SSTVModeSpec(
        "Robot 36",
        SSTVMode.ROBOT_36,
        0x08,
        320,
        240,
        150.0,
        9.0,
        3.0,
        "YUV",
        [88.0, 44.0, 44.0],
    ),
    0x0C: SSTVModeSpec(
        "Robot 72",
        SSTVMode.ROBOT_72,
        0x0C,
        320,
        240,
        300.0,
        9.0,
        3.0,
        "YUV",
        [138.0, 69.0, 69.0],
    ),
    # Scottie modes - GBR format
    0x3C: SSTVModeSpec(
        "Scottie 1",
        SSTVMode.SCOTTIE_1,
        0x3C,
        320,
        256,
        428.22,
        9.0,
        1.5,
        "GBR",
        [138.24, 138.24, 138.24],
    ),
    0x38: SSTVModeSpec(
        "Scottie 2",
        SSTVMode.SCOTTIE_2,
        0x38,
        320,
        256,
        277.69,
        9.0,
        1.5,
        "GBR",
        [88.064, 88.064, 88.064],
    ),
    0x4C: SSTVModeSpec(
        "Scottie DX",
        SSTVMode.SCOTTIE_DX,
        0x4C,
        320,
        256,
        1050.3,
        9.0,
        1.5,
        "GBR",
        [345.6, 345.6, 345.6],
    ),
    # Martin modes - GBR format
    0x2C: SSTVModeSpec(
        "Martin 1",
        SSTVMode.MARTIN_1,
        0x2C,
        320,
        256,
        446.446,
        4.862,
        0.572,
        "GBR",
        [146.432, 146.432, 146.432],
    ),
    0x28: SSTVModeSpec(
        "Martin 2",
        SSTVMode.MARTIN_2,
        0x28,
        320,
        256,
        226.798,
        4.862,
        0.572,
        "GBR",
        [73.216, 73.216, 73.216],
    ),
    # PD modes - YUV format (ISS favorites!)
    0x5D: SSTVModeSpec(
        "PD 90",
        SSTVMode.PD_90,
        0x5D,
        320,
        256,
        170.24,
        20.0,
        2.08,
        "YUV",
        [125.0, 125.0],
    ),
    0x5F: SSTVModeSpec(
        "PD 120",
        SSTVMode.PD_120,
        0x5F,
        640,
        496,
        121.6,
        20.0,
        2.08,
        "YUV",
        [190.0, 190.0],
    ),
    0x62: SSTVModeSpec(
        "PD 160", SSTVMode.PD_160, 0x62, 512, 400, 195.584, 20.0, 2.08, "YUV", [195.584]
    ),
    0x60: SSTVModeSpec(
        "PD 180",
        SSTVMode.PD_180,
        0x60,
        640,
        496,
        183.04,
        20.0,
        2.08,
        "YUV",
        [286.0, 286.0],
    ),
    0x61: SSTVModeSpec(
        "PD 240",
        SSTVMode.PD_240,
        0x61,
        640,
        496,
        244.48,
        20.0,
        2.08,
        "YUV",
        [382.0, 382.0],
    ),
    0x63: SSTVModeSpec(
        "PD 290",
        SSTVMode.PD_290,
        0x63,
        800,
        616,
        228.8,
        20.0,
        2.08,
        "YUV",
        [286.0, 286.0],
    ),
    # Wraase SC2
    0x37: SSTVModeSpec(
        "Wraase SC2-180",
        SSTVMode.WRAASE_SC2_180,
        0x37,
        320,
        256,
        711.04,
        5.5225,
        0.5,
        "RGB",
        [235.0, 235.0, 235.0],
    ),
}


@dataclass
class SSTVState:
    """Current state of SSTV decoding."""

    mode: Optional[SSTVModeSpec] = None
    is_receiving: bool = False
    current_line: int = 0
    line_buffer: List[float] = field(default_factory=list)
    image_data: Optional[np.ndarray] = None
    start_time: float = 0.0
    last_sync_time: float = 0.0
    signal_strength: float = 0.0
    vis_detected: bool = False
    complete: bool = False


class SSTVDecoder:
    """
    SSTV (Slow Scan Television) decoder.

    Decodes SSTV audio signals into images. Commonly used for receiving
    images from the International Space Station.

    Usage:
        decoder = SSTVDecoder(sample_rate=48000)

        # Feed audio samples (from FM demodulator output)
        decoder.process_audio(audio_samples)

        # Check if image is complete
        if decoder.is_complete():
            image = decoder.get_image()
            decoder.save_image("received_image.png")
    """

    # SSTV frequency constants (Hz)
    FREQ_BLACK = 1500  # Black level
    FREQ_WHITE = 2300  # White level
    FREQ_SYNC = 1200  # Sync pulse
    FREQ_VIS_BIT_0 = 1300  # VIS bit 0
    FREQ_VIS_BIT_1 = 1100  # VIS bit 1
    FREQ_LEADER = 1900  # Leader tone
    FREQ_BREAK = 1200  # Break tone

    # Timing constants (ms)
    VIS_BIT_TIME = 30.0  # Each VIS bit duration
    LEADER_TIME = 300.0  # Leader tone duration

    def __init__(self, sample_rate: float = 48000.0):
        """
        Initialize SSTV decoder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 48000)
        """
        self.sample_rate = sample_rate
        self.state = SSTVState()
        self._audio_buffer: List[float] = []
        self._freq_buffer: List[float] = []
        self._on_line_decoded: Optional[Callable[[int, np.ndarray], None]] = None
        self._on_image_complete: Optional[Callable[[np.ndarray], None]] = None
        self._on_mode_detected: Optional[Callable[[SSTVModeSpec], None]] = None

        # Goertzel filter parameters for frequency detection
        self._goertzel_n = int(sample_rate * 0.010)  # 10ms window

        logger.info(f"SSTVDecoder initialized at {sample_rate} Hz sample rate")

    def reset(self) -> None:
        """Reset decoder state for new reception."""
        self.state = SSTVState()
        self._audio_buffer.clear()
        self._freq_buffer.clear()
        logger.info("SSTV decoder reset")

    def set_on_line_decoded(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """Set callback for when each line is decoded."""
        self._on_line_decoded = callback

    def set_on_image_complete(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for when image is complete."""
        self._on_image_complete = callback

    def set_on_mode_detected(self, callback: Callable[[SSTVModeSpec], None]) -> None:
        """Set callback for when SSTV mode is detected."""
        self._on_mode_detected = callback

    def process_audio(self, samples: np.ndarray) -> None:
        """
        Process audio samples from FM demodulator.

        Args:
            samples: Audio samples (float32, mono)
        """
        # Convert to list and extend buffer
        self._audio_buffer.extend(samples.flatten().tolist())

        # Process in chunks
        chunk_size = int(self.sample_rate * 0.005)  # 5ms chunks

        while len(self._audio_buffer) >= chunk_size:
            chunk = np.array(self._audio_buffer[:chunk_size])
            self._audio_buffer = self._audio_buffer[chunk_size:]

            # Estimate frequency
            freq = self._estimate_frequency(chunk)
            self._freq_buffer.append(freq)

            # State machine
            if not self.state.vis_detected:
                self._detect_vis_header()
            elif self.state.is_receiving:
                self._decode_line_data(freq)

    def _estimate_frequency(self, samples: np.ndarray) -> float:
        """
        Estimate dominant frequency using Goertzel algorithm.

        Args:
            samples: Audio samples

        Returns:
            Estimated frequency in Hz
        """
        # Use zero-crossing method for quick estimate
        zero_crossings = np.where(np.diff(np.signbit(samples)))[0]

        if len(zero_crossings) < 2:
            return 0.0

        # Calculate average period between zero crossings
        periods = np.diff(zero_crossings)
        if len(periods) == 0:
            return 0.0

        avg_period = np.mean(periods)
        if avg_period == 0:
            return 0.0

        # Frequency = sample_rate / (2 * samples_per_half_cycle)
        freq = self.sample_rate / (2 * avg_period)

        return float(freq)

    def _detect_vis_header(self) -> None:
        """Detect VIS (Vertical Interval Signaling) header to identify mode."""
        if len(self._freq_buffer) < 100:  # Need enough samples
            return

        # Look for leader tone (1900 Hz for ~300ms)
        leader_count = sum(1 for f in self._freq_buffer[-50:] if 1850 < f < 1950)

        if leader_count > 40:  # ~80% match
            # Look for break tone (1200 Hz)
            recent = self._freq_buffer[-20:]
            break_count = sum(1 for f in recent if 1150 < f < 1250)

            if break_count > 5:
                # Try to decode VIS code
                vis_code = self._decode_vis_code()
                if vis_code is not None and vis_code in SSTV_MODES:
                    self.state.mode = SSTV_MODES[vis_code]
                    self.state.vis_detected = True
                    self.state.is_receiving = True
                    self.state.current_line = 0
                    self.state.start_time = time.time()

                    # Initialize image buffer
                    w, h = self.state.mode.width, self.state.mode.height
                    self.state.image_data = np.zeros((h, w, 3), dtype=np.uint8)

                    logger.info(f"SSTV mode detected: {self.state.mode.name} ({w}x{h})")

                    if self._on_mode_detected:
                        self._on_mode_detected(self.state.mode)

                    self._freq_buffer.clear()

    def _decode_vis_code(self) -> Optional[int]:
        """
        Decode VIS code from frequency buffer.

        Returns:
            VIS code byte or None if not detected
        """
        # VIS code is 8 bits, each 30ms, at 1100 Hz (1) or 1300 Hz (0)
        # Plus parity bit and start/stop bits

        if len(self._freq_buffer) < 30:
            return None

        # Look for the pattern in recent frequencies
        # This is a simplified detection - real implementation would be more robust
        bits = []

        # Sample the last portion of the buffer for VIS bits
        vis_samples = self._freq_buffer[-30:]

        for i in range(min(8, len(vis_samples))):
            freq = vis_samples[i]
            if 1050 < freq < 1150:
                bits.append(1)  # 1100 Hz = 1
            elif 1250 < freq < 1350:
                bits.append(0)  # 1300 Hz = 0
            else:
                return None

        if len(bits) < 8:
            return None

        # Convert bits to byte (LSB first)
        vis_code = sum(bit << i for i, bit in enumerate(bits[:8]))

        return vis_code if vis_code in SSTV_MODES else None

    def _decode_line_data(self, freq: float) -> None:
        """
        Decode pixel data from frequency.

        Args:
            freq: Current frequency estimate
        """
        if self.state.mode is None or self.state.image_data is None:
            return

        # Check for sync pulse
        if 1150 < freq < 1250:  # Sync frequency
            if self.state.line_buffer:
                self._finalize_line()
            return

        # Map frequency to luminance (1500-2300 Hz -> 0-255)
        if self.FREQ_BLACK <= freq <= self.FREQ_WHITE:
            value = (freq - self.FREQ_BLACK) / (self.FREQ_WHITE - self.FREQ_BLACK)
            value = max(0.0, min(1.0, value)) * 255.0
            self.state.line_buffer.append(value)

    def _finalize_line(self) -> None:
        """Process completed line and add to image."""
        if self.state.mode is None or self.state.image_data is None:
            return

        if len(self.state.line_buffer) < 10:
            self.state.line_buffer.clear()
            return

        mode = self.state.mode
        line_idx = self.state.current_line

        if line_idx >= mode.height:
            self._finalize_image()
            return

        # Resample line buffer to match image width
        line_data = np.array(self.state.line_buffer)

        # For RGB modes, split into channels
        if mode.color_format == "RGB":
            pixels_per_channel = len(line_data) // 3
            if pixels_per_channel > 0:
                r = self._resample_line(line_data[:pixels_per_channel], mode.width)
                g = self._resample_line(
                    line_data[pixels_per_channel : 2 * pixels_per_channel], mode.width
                )
                b = self._resample_line(line_data[2 * pixels_per_channel :], mode.width)

                self.state.image_data[line_idx, :, 0] = r
                self.state.image_data[line_idx, :, 1] = g
                self.state.image_data[line_idx, :, 2] = b

        elif mode.color_format == "GBR":
            pixels_per_channel = len(line_data) // 3
            if pixels_per_channel > 0:
                g = self._resample_line(line_data[:pixels_per_channel], mode.width)
                b = self._resample_line(
                    line_data[pixels_per_channel : 2 * pixels_per_channel], mode.width
                )
                r = self._resample_line(line_data[2 * pixels_per_channel :], mode.width)

                self.state.image_data[line_idx, :, 0] = r
                self.state.image_data[line_idx, :, 1] = g
                self.state.image_data[line_idx, :, 2] = b

        elif mode.color_format == "YUV":
            # YUV to RGB conversion
            pixels_per_channel = (
                len(line_data) // 2
                if len(mode.channel_times_ms) == 2
                else len(line_data)
            )
            if pixels_per_channel > 0:
                y = self._resample_line(line_data[:pixels_per_channel], mode.width)

                if (
                    len(mode.channel_times_ms) >= 2
                    and len(line_data) >= 2 * pixels_per_channel
                ):
                    # Has chroma
                    uv_data = line_data[pixels_per_channel:]
                    u = self._resample_line(uv_data[: len(uv_data) // 2], mode.width)
                    v = self._resample_line(uv_data[len(uv_data) // 2 :], mode.width)

                    # YUV to RGB
                    r, g, b = self._yuv_to_rgb(y, u, v)
                else:
                    # Grayscale
                    r = g = b = y

                self.state.image_data[line_idx, :, 0] = r
                self.state.image_data[line_idx, :, 1] = g
                self.state.image_data[line_idx, :, 2] = b

        # Callback
        if self._on_line_decoded:
            self._on_line_decoded(line_idx, self.state.image_data[line_idx])

        self.state.current_line += 1
        self.state.line_buffer.clear()

        # Check if image is complete
        if self.state.current_line >= mode.height:
            self._finalize_image()

    def _resample_line(self, data: np.ndarray, target_width: int) -> np.ndarray:
        """Resample line data to target width."""
        if len(data) == 0:
            return np.zeros(target_width, dtype=np.uint8)

        if len(data) == target_width:
            return data.astype(np.uint8)

        # Linear interpolation
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_width)
        resampled = np.interp(x_new, x_old, data)

        return np.clip(resampled, 0, 255).astype(np.uint8)

    def _yuv_to_rgb(
        self, y: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert YUV to RGB."""
        # Center U and V around 128
        u = u.astype(np.float32) - 128
        v = v.astype(np.float32) - 128
        y = y.astype(np.float32)

        # YUV to RGB conversion
        r = y + 1.402 * v
        g = y - 0.344 * u - 0.714 * v
        b = y + 1.772 * u

        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

        return r, g, b

    def _finalize_image(self) -> None:
        """Mark image as complete."""
        self.state.complete = True
        self.state.is_receiving = False

        duration = time.time() - self.state.start_time
        logger.info(
            f"SSTV image complete: {self.state.current_line} lines in {duration:.1f}s"
        )

        if self._on_image_complete and self.state.image_data is not None:
            self._on_image_complete(self.state.image_data)

    def is_receiving(self) -> bool:
        """Check if currently receiving an image."""
        return self.state.is_receiving

    def is_complete(self) -> bool:
        """Check if image reception is complete."""
        return self.state.complete

    def get_progress(self) -> float:
        """
        Get reception progress.

        Returns:
            Progress as fraction 0.0 to 1.0
        """
        if self.state.mode is None:
            return 0.0
        return self.state.current_line / self.state.mode.height

    def get_image(self) -> Optional[np.ndarray]:
        """
        Get the decoded image.

        Returns:
            RGB image as numpy array (H, W, 3) or None
        """
        return self.state.image_data

    def get_mode(self) -> Optional[SSTVModeSpec]:
        """Get detected SSTV mode."""
        return self.state.mode

    def get_status(self) -> dict:
        """Get decoder status."""
        return {
            "mode": self.state.mode.name if self.state.mode else None,
            "is_receiving": self.state.is_receiving,
            "complete": self.state.complete,
            "current_line": self.state.current_line,
            "total_lines": self.state.mode.height if self.state.mode else 0,
            "progress": self.get_progress(),
            "signal_strength": self.state.signal_strength,
        }

    def save_image(self, path: str) -> bool:
        """
        Save decoded image to file.

        Args:
            path: Output file path (PNG, JPG, etc.)

        Returns:
            True if saved successfully
        """
        if self.state.image_data is None:
            logger.error("No image data to save")
            return False

        try:
            # Try PIL first
            try:
                from PIL import Image

                img = Image.fromarray(self.state.image_data, "RGB")
                img.save(path)
                logger.info(f"Saved SSTV image to {path}")
                return True
            except ImportError:
                pass

            # Fallback to matplotlib
            try:
                import matplotlib.pyplot as plt

                plt.imsave(path, self.state.image_data)
                logger.info(f"Saved SSTV image to {path}")
                return True
            except ImportError:
                pass

            # Fallback to raw numpy save
            np.save(
                path.replace(".png", ".npy").replace(".jpg", ".npy"),
                self.state.image_data,
            )
            logger.info(
                f"Saved SSTV image data to {path}.npy (install PIL for image format)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False


class SSTVImageViewer:
    """
    Simple SSTV image viewer/manager.

    Manages received SSTV images with history and metadata.
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize viewer.

        Args:
            save_dir: Directory for auto-saving images
        """
        self.save_dir = Path(save_dir) if save_dir else Path("sstv_images")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.images: List[dict] = []
        self.current_index: int = -1

    def add_image(
        self, image: np.ndarray, mode: SSTVModeSpec, auto_save: bool = True
    ) -> int:
        """
        Add a received image.

        Args:
            image: Image data as numpy array
            mode: SSTV mode used
            auto_save: Whether to auto-save to disk

        Returns:
            Image index
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        entry = {
            "image": image.copy(),
            "mode": mode.name,
            "timestamp": timestamp,
            "width": mode.width,
            "height": mode.height,
            "filename": None,
        }

        if auto_save:
            filename = f"sstv_{timestamp}_{mode.name.replace(' ', '_')}.png"
            filepath = self.save_dir / filename

            try:
                from PIL import Image

                img = Image.fromarray(image, "RGB")
                img.save(str(filepath))
                entry["filename"] = str(filepath)
                logger.info(f"Auto-saved SSTV image: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to auto-save: {e}")

        self.images.append(entry)
        self.current_index = len(self.images) - 1

        return self.current_index

    def get_current_image(self) -> Optional[np.ndarray]:
        """Get current image."""
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]["image"]
        return None

    def get_image_count(self) -> int:
        """Get number of stored images."""
        return len(self.images)

    def next_image(self) -> Optional[np.ndarray]:
        """Go to next image."""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            return self.get_current_image()
        return None

    def prev_image(self) -> Optional[np.ndarray]:
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            return self.get_current_image()
        return None

    def get_image_info(self, index: Optional[int] = None) -> Optional[dict]:
        """Get info for an image."""
        idx = index if index is not None else self.current_index
        if 0 <= idx < len(self.images):
            info = self.images[idx].copy()
            del info["image"]  # Don't include raw data
            return info
        return None

    def clear(self) -> None:
        """Clear all stored images."""
        self.images.clear()
        self.current_index = -1


__all__ = [
    "SSTVMode",
    "SSTVModeSpec",
    "SSTV_MODES",
    "SSTVState",
    "SSTVDecoder",
    "SSTVImageViewer",
]

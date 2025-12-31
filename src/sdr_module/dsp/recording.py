"""
I/Q Recording and Playback for SDR signals.

Provides functionality to record and playback raw baseband signals:
- Multiple file formats: raw I/Q, WAV, SigMF
- Various sample formats: 8-bit, 16-bit, 32-bit float
- Streaming recording with buffering
- Metadata support (frequency, sample rate, timestamps)

Also provides Audio Recording and Playback for demodulated audio:
- Standard audio sample rates: 8kHz, 22.05kHz, 44.1kHz, 48kHz
- WAV format with 8-bit, 16-bit, 24-bit, 32-bit float support
- Streaming audio capture from demodulators
"""

import json
import struct
import wave
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import numpy as np


class SampleFormat(Enum):
    """Sample format for I/Q data."""

    UINT8 = "cu8"  # Unsigned 8-bit (RTL-SDR native)
    INT8 = "cs8"  # Signed 8-bit
    INT16 = "cs16"  # Signed 16-bit (common)
    FLOAT32 = "cf32"  # 32-bit float complex
    FLOAT64 = "cf64"  # 64-bit float complex


class FileFormat(Enum):
    """Recording file format."""

    RAW = "raw"  # Raw I/Q binary
    WAV = "wav"  # WAV file (audio compatible)
    SIGMF = "sigmf"  # Signal Metadata Format


@dataclass
class RecordingMetadata:
    """Metadata for I/Q recording."""

    # Core parameters
    sample_rate: float = 0.0
    center_frequency: float = 0.0
    sample_format: SampleFormat = SampleFormat.FLOAT32

    # Optional parameters
    bandwidth: float = 0.0
    gain: float = 0.0
    antenna: str = ""

    # Recording info
    start_time: str = ""
    duration_seconds: float = 0.0
    num_samples: int = 0

    # Hardware info
    hardware: str = ""
    description: str = ""

    # Custom fields
    custom: Dict[str, Any] = field(default_factory=dict)


class IQRecorder:
    """
    I/Q signal recorder for SDR applications.

    Records raw baseband I/Q samples to files for later analysis
    or playback. Supports multiple file formats and sample types.

    Features:
    - Multiple formats: raw binary, WAV, SigMF
    - Sample formats: 8-bit, 16-bit, 32-bit float
    - Streaming recording with buffering
    - Automatic metadata generation
    - Timestamp support

    Example:
        recorder = IQRecorder(
            sample_rate=2.4e6,
            center_frequency=100e6,
            sample_format=SampleFormat.INT16
        )
        recorder.start("recording.raw")
        recorder.write(samples)
        recorder.stop()
    """

    def __init__(
        self,
        sample_rate: float,
        center_frequency: float = 0.0,
        sample_format: SampleFormat = SampleFormat.FLOAT32,
        file_format: FileFormat = FileFormat.RAW,
        buffer_size: int = 65536,
    ):
        """
        Initialize I/Q recorder.

        Args:
            sample_rate: Sample rate in Hz
            center_frequency: Center frequency in Hz
            sample_format: Sample format for storage
            file_format: File format to use
            buffer_size: Internal buffer size in samples
        """
        self._sample_rate = sample_rate
        self._center_frequency = center_frequency
        self._sample_format = sample_format
        self._file_format = file_format
        self._buffer_size = buffer_size

        # Recording state
        self._file: Optional[BinaryIO] = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._filepath: Optional[Path] = None
        self._recording = False
        self._samples_written = 0
        self._start_time: Optional[datetime] = None

        # Buffer for streaming
        self._buffer = np.array([], dtype=np.complex64)

        # Metadata
        self._metadata = RecordingMetadata(
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            sample_format=sample_format,
        )

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def center_frequency(self) -> float:
        """Get center frequency."""
        return self._center_frequency

    @property
    def sample_format(self) -> SampleFormat:
        """Get sample format."""
        return self._sample_format

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def samples_written(self) -> int:
        """Get number of samples written."""
        return self._samples_written

    @property
    def duration_seconds(self) -> float:
        """Get recording duration in seconds."""
        return self._samples_written / self._sample_rate

    @property
    def file_size_bytes(self) -> int:
        """Get estimated file size in bytes."""
        bytes_per_sample = self._get_bytes_per_sample()
        return self._samples_written * bytes_per_sample * 2  # *2 for I and Q

    def _get_bytes_per_sample(self) -> int:
        """Get bytes per sample component."""
        if self._sample_format == SampleFormat.UINT8:
            return 1
        elif self._sample_format == SampleFormat.INT8:
            return 1
        elif self._sample_format == SampleFormat.INT16:
            return 2
        elif self._sample_format == SampleFormat.FLOAT32:
            return 4
        elif self._sample_format == SampleFormat.FLOAT64:
            return 8
        return 4

    def _get_numpy_dtype(self) -> np.dtype:
        """Get numpy dtype for sample format."""
        if self._sample_format == SampleFormat.UINT8:
            return np.uint8
        elif self._sample_format == SampleFormat.INT8:
            return np.int8
        elif self._sample_format == SampleFormat.INT16:
            return np.int16
        elif self._sample_format == SampleFormat.FLOAT32:
            return np.float32
        elif self._sample_format == SampleFormat.FLOAT64:
            return np.float64
        return np.float32

    def _convert_samples(self, samples: np.ndarray) -> np.ndarray:
        """Convert samples to target format."""
        # Ensure complex
        if not np.iscomplexobj(samples):
            samples = samples.astype(np.complex64)

        # Interleave I/Q
        interleaved = np.zeros(len(samples) * 2, dtype=np.float64)
        interleaved[0::2] = samples.real
        interleaved[1::2] = samples.imag

        # Scale and convert to target format
        if self._sample_format == SampleFormat.UINT8:
            # Scale to 0-255, center at 127.5
            scaled = np.clip(interleaved * 127.5 + 127.5, 0, 255)
            return scaled.astype(np.uint8)

        elif self._sample_format == SampleFormat.INT8:
            # Scale to -128 to 127
            scaled = np.clip(interleaved * 127, -128, 127)
            return scaled.astype(np.int8)

        elif self._sample_format == SampleFormat.INT16:
            # Scale to -32768 to 32767
            scaled = np.clip(interleaved * 32767, -32768, 32767)
            return scaled.astype(np.int16)

        elif self._sample_format == SampleFormat.FLOAT32:
            return interleaved.astype(np.float32)

        elif self._sample_format == SampleFormat.FLOAT64:
            return interleaved.astype(np.float64)

        return interleaved.astype(np.float32)

    def set_metadata(self, **kwargs) -> None:
        """
        Set additional metadata.

        Args:
            **kwargs: Metadata fields to set
        """
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)
            else:
                self._metadata.custom[key] = value

    def start(self, filepath: Union[str, Path]) -> None:
        """
        Start recording to file.

        Args:
            filepath: Output file path
        """
        if self._recording:
            raise RuntimeError("Already recording")

        self._filepath = Path(filepath)
        self._samples_written = 0
        self._start_time = datetime.utcnow()
        self._metadata.start_time = self._start_time.isoformat() + "Z"

        if self._file_format == FileFormat.WAV:
            self._start_wav()
        else:
            self._file = open(self._filepath, "wb")

        self._recording = True

    def _start_wav(self) -> None:
        """Start WAV file recording."""
        self._wav_file = wave.open(str(self._filepath), "wb")

        # WAV parameters
        # Use 2 channels for I/Q (stereo)
        n_channels = 2

        # Sample width based on format
        if self._sample_format in (SampleFormat.UINT8, SampleFormat.INT8):
            sample_width = 1
        elif self._sample_format == SampleFormat.INT16:
            sample_width = 2
        else:
            sample_width = 2  # Default to 16-bit for WAV

        # Limit sample rate for WAV compatibility
        wav_sample_rate = min(int(self._sample_rate), 192000)

        self._wav_file.setnchannels(n_channels)
        self._wav_file.setsampwidth(sample_width)
        self._wav_file.setframerate(wav_sample_rate)

    def write(self, samples: np.ndarray) -> int:
        """
        Write samples to recording.

        Args:
            samples: Complex samples to write

        Returns:
            Number of samples written
        """
        if not self._recording:
            raise RuntimeError("Not recording")

        # Convert samples
        converted = self._convert_samples(samples)

        if self._file_format == FileFormat.WAV:
            self._write_wav(converted)
        elif self._file is not None:
            converted.tofile(self._file)

        self._samples_written += len(samples)
        return len(samples)

    def _write_wav(self, data: np.ndarray) -> None:
        """Write data to WAV file."""
        if self._wav_file is None:
            return

        # Convert to bytes
        if data.dtype == np.int16:
            self._wav_file.writeframes(data.tobytes())
        elif data.dtype in (np.uint8, np.int8):
            self._wav_file.writeframes(data.tobytes())
        else:
            # Convert float to int16 for WAV
            int_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
            self._wav_file.writeframes(int_data.tobytes())

    def stop(self) -> RecordingMetadata:
        """
        Stop recording and finalize file.

        Returns:
            Recording metadata
        """
        if not self._recording:
            raise RuntimeError("Not recording")

        # Update metadata
        self._metadata.num_samples = self._samples_written
        self._metadata.duration_seconds = self.duration_seconds

        # Close files
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

        if self._file is not None:
            self._file.close()
            self._file = None

        # Write SigMF metadata if using that format
        if self._file_format == FileFormat.SIGMF:
            self._write_sigmf_metadata()

        self._recording = False
        return self._metadata

    def _write_sigmf_metadata(self) -> None:
        """Write SigMF metadata file."""
        if self._filepath is None:
            return

        # SigMF format type string
        format_map = {
            SampleFormat.UINT8: "cu8",
            SampleFormat.INT8: "ci8",
            SampleFormat.INT16: "ci16_le",
            SampleFormat.FLOAT32: "cf32_le",
            SampleFormat.FLOAT64: "cf64_le",
        }

        sigmf_meta = {
            "global": {
                "core:datatype": format_map.get(self._sample_format, "cf32_le"),
                "core:sample_rate": self._sample_rate,
                "core:version": "1.0.0",
                "core:description": self._metadata.description or "SDR Recording",
                "core:author": "",
                "core:recorder": "SDR Module IQRecorder",
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": self._center_frequency,
                    "core:datetime": self._metadata.start_time,
                }
            ],
            "annotations": [],
        }

        # Add optional fields
        if self._metadata.hardware:
            sigmf_meta["global"]["core:hw"] = self._metadata.hardware

        # Write metadata file
        meta_path = self._filepath.with_suffix(".sigmf-meta")
        with open(meta_path, "w") as f:
            json.dump(sigmf_meta, f, indent=2)

        # Rename data file for SigMF
        data_path = self._filepath.with_suffix(".sigmf-data")
        if self._filepath.exists() and self._filepath != data_path:
            self._filepath.rename(data_path)

    def get_metadata(self) -> RecordingMetadata:
        """Get current metadata."""
        return self._metadata


class IQPlayer:
    """
    I/Q signal playback from recorded files.

    Reads recorded I/Q files and provides samples for processing
    or transmission.

    Features:
    - Supports raw, WAV, and SigMF formats
    - Automatic format detection
    - Streaming playback with buffering
    - Loop and seek support

    Example:
        player = IQPlayer("recording.sigmf-data")
        while not player.eof:
            samples = player.read(1024)
            process(samples)
        player.close()
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        sample_format: Optional[SampleFormat] = None,
        sample_rate: Optional[float] = None,
    ):
        """
        Initialize I/Q player.

        Args:
            filepath: Path to recording file
            sample_format: Sample format (auto-detect if None)
            sample_rate: Sample rate (auto-detect if None)
        """
        self._filepath = Path(filepath)
        self._sample_format = sample_format
        self._sample_rate = sample_rate or 0.0
        self._center_frequency = 0.0

        # Playback state
        self._file: Optional[BinaryIO] = None
        self._wav_file: Optional[wave.Wave_read] = None
        self._position = 0
        self._total_samples = 0
        self._eof = False

        # Metadata
        self._metadata: Optional[RecordingMetadata] = None

        # Open file
        self._open()

    def _open(self) -> None:
        """Open the recording file."""
        suffix = self._filepath.suffix.lower()

        # Check for SigMF
        if suffix in (".sigmf-data", ".sigmf"):
            self._open_sigmf()
        elif suffix == ".wav":
            self._open_wav()
        else:
            self._open_raw()

    def _open_sigmf(self) -> None:
        """Open SigMF recording."""
        # Find metadata file
        meta_path = self._filepath.with_suffix(".sigmf-meta")
        data_path = self._filepath.with_suffix(".sigmf-data")

        if not data_path.exists():
            data_path = self._filepath

        # Read metadata
        if meta_path.exists():
            with open(meta_path, "r") as f:
                sigmf_meta = json.load(f)

            global_meta = sigmf_meta.get("global", {})
            captures = sigmf_meta.get("captures", [{}])

            # Parse datatype
            datatype = global_meta.get("core:datatype", "cf32_le")
            format_map = {
                "cu8": SampleFormat.UINT8,
                "ci8": SampleFormat.INT8,
                "ci16_le": SampleFormat.INT16,
                "cf32_le": SampleFormat.FLOAT32,
                "cf64_le": SampleFormat.FLOAT64,
            }
            self._sample_format = format_map.get(datatype, SampleFormat.FLOAT32)

            self._sample_rate = global_meta.get("core:sample_rate", 0.0)
            if captures:
                self._center_frequency = captures[0].get("core:frequency", 0.0)

            self._metadata = RecordingMetadata(
                sample_rate=self._sample_rate,
                center_frequency=self._center_frequency,
                sample_format=self._sample_format,
                description=global_meta.get("core:description", ""),
                hardware=global_meta.get("core:hw", ""),
            )

        # Open data file
        self._file = open(data_path, "rb")

        # Calculate total samples
        file_size = data_path.stat().st_size
        bytes_per_sample = self._get_bytes_per_sample() * 2  # I and Q
        self._total_samples = file_size // bytes_per_sample

    def _open_wav(self) -> None:
        """Open WAV recording."""
        self._wav_file = wave.open(str(self._filepath), "rb")

        # Get parameters
        self._wav_file.getnchannels()
        sample_width = self._wav_file.getsampwidth()
        self._sample_rate = float(self._wav_file.getframerate())
        self._total_samples = self._wav_file.getnframes()

        # Determine sample format
        if sample_width == 1:
            self._sample_format = SampleFormat.UINT8
        elif sample_width == 2:
            self._sample_format = SampleFormat.INT16
        else:
            self._sample_format = SampleFormat.INT16

        self._metadata = RecordingMetadata(
            sample_rate=self._sample_rate,
            sample_format=self._sample_format,
            num_samples=self._total_samples,
        )

    def _open_raw(self) -> None:
        """Open raw I/Q file."""
        # Try to detect format from extension
        suffix = self._filepath.suffix.lower()

        format_map = {
            ".cu8": SampleFormat.UINT8,
            ".cs8": SampleFormat.INT8,
            ".cs16": SampleFormat.INT16,
            ".cf32": SampleFormat.FLOAT32,
            ".cf64": SampleFormat.FLOAT64,
            ".raw": SampleFormat.FLOAT32,
            ".iq": SampleFormat.FLOAT32,
        }

        if self._sample_format is None:
            self._sample_format = format_map.get(suffix, SampleFormat.FLOAT32)

        self._file = open(self._filepath, "rb")

        # Calculate total samples
        file_size = self._filepath.stat().st_size
        bytes_per_sample = self._get_bytes_per_sample() * 2
        self._total_samples = file_size // bytes_per_sample

        self._metadata = RecordingMetadata(
            sample_rate=self._sample_rate,
            sample_format=self._sample_format,
            num_samples=self._total_samples,
        )

    def _get_bytes_per_sample(self) -> int:
        """Get bytes per sample component."""
        if self._sample_format == SampleFormat.UINT8:
            return 1
        elif self._sample_format == SampleFormat.INT8:
            return 1
        elif self._sample_format == SampleFormat.INT16:
            return 2
        elif self._sample_format == SampleFormat.FLOAT32:
            return 4
        elif self._sample_format == SampleFormat.FLOAT64:
            return 8
        return 4

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def center_frequency(self) -> float:
        """Get center frequency."""
        return self._center_frequency

    @property
    def sample_format(self) -> SampleFormat:
        """Get sample format."""
        return self._sample_format

    @property
    def total_samples(self) -> int:
        """Get total number of samples."""
        return self._total_samples

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if self._sample_rate > 0:
            return self._total_samples / self._sample_rate
        return 0.0

    @property
    def position(self) -> int:
        """Get current sample position."""
        return self._position

    @property
    def eof(self) -> bool:
        """Check if at end of file."""
        return self._eof

    @property
    def metadata(self) -> Optional[RecordingMetadata]:
        """Get recording metadata."""
        return self._metadata

    def read(self, num_samples: int) -> np.ndarray:
        """
        Read samples from recording.

        Args:
            num_samples: Number of samples to read

        Returns:
            Complex samples array
        """
        if self._wav_file is not None:
            return self._read_wav(num_samples)
        elif self._file is not None:
            return self._read_raw(num_samples)
        return np.array([], dtype=np.complex64)

    def _read_raw(self, num_samples: int) -> np.ndarray:
        """Read from raw file."""
        bytes_per_sample = self._get_bytes_per_sample()
        bytes_to_read = num_samples * bytes_per_sample * 2

        data = self._file.read(bytes_to_read)

        if len(data) == 0:
            self._eof = True
            return np.array([], dtype=np.complex64)

        # Convert to numpy
        dtype = self._get_numpy_dtype()
        samples = np.frombuffer(data, dtype=dtype)

        # De-interleave I/Q
        if len(samples) >= 2:
            n_complex = len(samples) // 2
            i_samples = samples[0::2].astype(np.float64)
            q_samples = samples[1::2].astype(np.float64)

            # Normalize based on format
            if self._sample_format == SampleFormat.UINT8:
                i_samples = (i_samples - 127.5) / 127.5
                q_samples = (q_samples - 127.5) / 127.5
            elif self._sample_format == SampleFormat.INT8:
                i_samples = i_samples / 127.0
                q_samples = q_samples / 127.0
            elif self._sample_format == SampleFormat.INT16:
                i_samples = i_samples / 32767.0
                q_samples = q_samples / 32767.0

            complex_samples = i_samples + 1j * q_samples
            self._position += n_complex
            return complex_samples.astype(np.complex64)

        return np.array([], dtype=np.complex64)

    def _get_numpy_dtype(self) -> np.dtype:
        """Get numpy dtype."""
        if self._sample_format == SampleFormat.UINT8:
            return np.uint8
        elif self._sample_format == SampleFormat.INT8:
            return np.int8
        elif self._sample_format == SampleFormat.INT16:
            return np.int16
        elif self._sample_format == SampleFormat.FLOAT32:
            return np.float32
        elif self._sample_format == SampleFormat.FLOAT64:
            return np.float64
        return np.float32

    def _read_wav(self, num_samples: int) -> np.ndarray:
        """Read from WAV file."""
        frames = self._wav_file.readframes(num_samples)

        if len(frames) == 0:
            self._eof = True
            return np.array([], dtype=np.complex64)

        # Convert bytes to samples
        sample_width = self._wav_file.getsampwidth()
        n_channels = self._wav_file.getnchannels()

        if sample_width == 1:
            samples = np.frombuffer(frames, dtype=np.uint8)
            samples = (samples.astype(np.float64) - 128) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(frames, dtype=np.int16)
            samples = samples.astype(np.float64) / 32767.0
        else:
            samples = np.frombuffer(frames, dtype=np.int16)
            samples = samples.astype(np.float64) / 32767.0

        # Handle channels
        if n_channels == 2:
            # Stereo: I on left, Q on right
            len(samples) // 2
            i_samples = samples[0::2]
            q_samples = samples[1::2]
            complex_samples = i_samples + 1j * q_samples
        else:
            # Mono: real only
            complex_samples = samples.astype(np.complex64)

        self._position += len(complex_samples)
        return complex_samples.astype(np.complex64)

    def seek(self, position: int) -> None:
        """
        Seek to sample position.

        Args:
            position: Sample position
        """
        if position < 0:
            position = 0
        if position >= self._total_samples:
            position = self._total_samples - 1

        if self._wav_file is not None:
            self._wav_file.setpos(position)
        elif self._file is not None:
            bytes_per_sample = self._get_bytes_per_sample() * 2
            self._file.seek(position * bytes_per_sample)

        self._position = position
        self._eof = False

    def rewind(self) -> None:
        """Rewind to beginning."""
        self.seek(0)

    def close(self) -> None:
        """Close the file."""
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class RecordingSession:
    """
    Managed recording session with automatic file naming and metadata.

    Example:
        with RecordingSession(sample_rate=2.4e6, output_dir="recordings") as session:
            session.set_frequency(100e6)
            session.record(samples)
    """

    def __init__(
        self,
        sample_rate: float,
        output_dir: Union[str, Path] = ".",
        file_format: FileFormat = FileFormat.SIGMF,
        sample_format: SampleFormat = SampleFormat.INT16,
        prefix: str = "recording",
    ):
        """
        Initialize recording session.

        Args:
            sample_rate: Sample rate in Hz
            output_dir: Output directory
            file_format: File format to use
            sample_format: Sample format
            prefix: Filename prefix
        """
        self._sample_rate = sample_rate
        self._output_dir = Path(output_dir)
        self._file_format = file_format
        self._sample_format = sample_format
        self._prefix = prefix

        self._recorder: Optional[IQRecorder] = None
        self._current_file: Optional[Path] = None
        self._recording_count = 0

        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self) -> Path:
        """Generate unique filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        ext_map = {
            FileFormat.RAW: ".raw",
            FileFormat.WAV: ".wav",
            FileFormat.SIGMF: ".sigmf-data",
        }
        ext = ext_map.get(self._file_format, ".raw")

        filename = f"{self._prefix}_{timestamp}{ext}"
        return self._output_dir / filename

    def start(self, center_frequency: float = 0.0) -> Path:
        """
        Start new recording.

        Args:
            center_frequency: Center frequency in Hz

        Returns:
            Path to recording file
        """
        if self._recorder is not None and self._recorder.is_recording:
            self.stop()

        filepath = self._generate_filename()

        self._recorder = IQRecorder(
            sample_rate=self._sample_rate,
            center_frequency=center_frequency,
            sample_format=self._sample_format,
            file_format=self._file_format,
        )

        self._recorder.start(filepath)
        self._current_file = filepath
        self._recording_count += 1

        return filepath

    def record(self, samples: np.ndarray) -> int:
        """
        Record samples (starts recording if not already).

        Args:
            samples: Samples to record

        Returns:
            Number of samples written
        """
        if self._recorder is None or not self._recorder.is_recording:
            self.start()

        return self._recorder.write(samples)

    def stop(self) -> Optional[RecordingMetadata]:
        """
        Stop current recording.

        Returns:
            Recording metadata
        """
        if self._recorder is not None and self._recorder.is_recording:
            return self._recorder.stop()
        return None

    @property
    def is_recording(self) -> bool:
        """Check if recording."""
        return self._recorder is not None and self._recorder.is_recording

    @property
    def current_file(self) -> Optional[Path]:
        """Get current recording file path."""
        return self._current_file

    @property
    def recording_count(self) -> int:
        """Get number of recordings made."""
        return self._recording_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def load_iq_file(
    filepath: Union[str, Path],
    sample_format: Optional[SampleFormat] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, RecordingMetadata]:
    """
    Load entire I/Q file into memory.

    Args:
        filepath: Path to I/Q file
        sample_format: Sample format (auto-detect if None)
        max_samples: Maximum samples to load

    Returns:
        Tuple of (samples, metadata)
    """
    with IQPlayer(filepath, sample_format) as player:
        if max_samples is None:
            max_samples = player.total_samples

        samples = player.read(max_samples)
        metadata = player.metadata

    return samples, metadata


def save_iq_file(
    filepath: Union[str, Path],
    samples: np.ndarray,
    sample_rate: float,
    center_frequency: float = 0.0,
    sample_format: SampleFormat = SampleFormat.FLOAT32,
    file_format: FileFormat = FileFormat.RAW,
) -> RecordingMetadata:
    """
    Save samples to I/Q file.

    Args:
        filepath: Output file path
        samples: Complex samples to save
        sample_rate: Sample rate in Hz
        center_frequency: Center frequency in Hz
        sample_format: Sample format
        file_format: File format

    Returns:
        Recording metadata
    """
    recorder = IQRecorder(
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        sample_format=sample_format,
        file_format=file_format,
    )

    recorder.start(filepath)
    recorder.write(samples)
    return recorder.stop()


class AudioSampleFormat(Enum):
    """Sample format for audio data."""

    UINT8 = "u8"  # Unsigned 8-bit
    INT16 = "s16"  # Signed 16-bit (CD quality)
    INT24 = "s24"  # Signed 24-bit (studio quality)
    FLOAT32 = "f32"  # 32-bit float


class AudioSampleRate(Enum):
    """Standard audio sample rates."""

    RATE_8000 = 8000  # Telephone quality
    RATE_11025 = 11025  # Low quality
    RATE_22050 = 22050  # FM radio quality
    RATE_44100 = 44100  # CD quality
    RATE_48000 = 48000  # Professional audio


@dataclass
class AudioMetadata:
    """Metadata for audio recording."""

    # Core parameters
    sample_rate: int = 44100
    sample_format: AudioSampleFormat = AudioSampleFormat.INT16
    channels: int = 1  # 1=mono, 2=stereo

    # Recording info
    start_time: str = ""
    duration_seconds: float = 0.0
    num_samples: int = 0

    # Source info
    source_frequency: float = 0.0  # Original tuned frequency
    demodulation: str = ""  # Demodulation mode (FM, AM, SSB, etc.)

    # Optional
    description: str = ""
    custom: Dict[str, Any] = field(default_factory=dict)


class AudioRecorder:
    """
    Audio recorder for demodulated SDR signals.

    Records real audio samples (not complex I/Q) to WAV files.
    Designed for capturing demodulated audio from FM, AM, SSB, etc.

    Features:
    - Standard audio sample rates: 8kHz to 48kHz
    - Multiple bit depths: 8-bit, 16-bit, 24-bit, 32-bit float
    - WAV format for universal compatibility
    - Streaming recording with minimal latency
    - Automatic normalization and clipping protection

    Example:
        recorder = AudioRecorder(sample_rate=44100)
        recorder.start("audio.wav")
        recorder.write(demodulated_samples)
        recorder.stop()
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        sample_format: AudioSampleFormat = AudioSampleFormat.INT16,
        channels: int = 1,
        normalize: bool = True,
    ):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz (8000-48000)
            sample_format: Sample format for storage
            channels: Number of channels (1=mono, 2=stereo)
            normalize: Auto-normalize input to prevent clipping
        """
        self._sample_rate = sample_rate
        self._sample_format = sample_format
        self._channels = channels
        self._normalize = normalize

        # Recording state
        self._wav_file: Optional[wave.Wave_write] = None
        self._filepath: Optional[Path] = None
        self._recording = False
        self._samples_written = 0
        self._start_time: Optional[datetime] = None

        # Peak tracking for normalization
        self._peak_level = 0.0
        self._auto_gain = 1.0

        # Metadata
        self._metadata = AudioMetadata(
            sample_rate=sample_rate, sample_format=sample_format, channels=channels
        )

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @property
    def sample_format(self) -> AudioSampleFormat:
        """Get sample format."""
        return self._sample_format

    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self._channels

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def samples_written(self) -> int:
        """Get number of samples written."""
        return self._samples_written

    @property
    def duration_seconds(self) -> float:
        """Get recording duration in seconds."""
        return self._samples_written / self._sample_rate

    @property
    def peak_level(self) -> float:
        """Get peak level seen during recording."""
        return self._peak_level

    def _get_sample_width(self) -> int:
        """Get sample width in bytes."""
        if self._sample_format == AudioSampleFormat.UINT8:
            return 1
        elif self._sample_format == AudioSampleFormat.INT16:
            return 2
        elif self._sample_format == AudioSampleFormat.INT24:
            return 3
        elif self._sample_format == AudioSampleFormat.FLOAT32:
            return 4
        return 2

    def _convert_samples(self, samples: np.ndarray) -> bytes:
        """Convert float samples to target format."""
        # Ensure 1D real array
        if np.iscomplexobj(samples):
            samples = samples.real
        samples = samples.flatten()

        # Track peak
        peak = np.max(np.abs(samples)) if len(samples) > 0 else 0.0
        self._peak_level = max(self._peak_level, peak)

        # Normalize if needed
        if self._normalize and peak > 0:
            if peak > 1.0:
                samples = samples / peak * 0.99

        # Clip to valid range
        samples = np.clip(samples, -1.0, 1.0)

        # Convert to target format
        if self._sample_format == AudioSampleFormat.UINT8:
            # Scale to 0-255, center at 128
            int_samples = ((samples + 1.0) * 127.5).astype(np.uint8)
            return int_samples.tobytes()

        elif self._sample_format == AudioSampleFormat.INT16:
            # Scale to -32768 to 32767
            int_samples = (samples * 32767).astype(np.int16)
            return int_samples.tobytes()

        elif self._sample_format == AudioSampleFormat.INT24:
            # Scale to 24-bit range
            int_samples = (samples * 8388607).astype(np.int32)
            # Pack as 3 bytes per sample (little-endian)
            result = bytearray()
            for s in int_samples:
                result.extend(struct.pack("<i", s)[:3])
            return bytes(result)

        elif self._sample_format == AudioSampleFormat.FLOAT32:
            # WAV float format (rarely used, convert to int16)
            int_samples = (samples * 32767).astype(np.int16)
            return int_samples.tobytes()

        return b""

    def set_metadata(self, **kwargs) -> None:
        """
        Set additional metadata.

        Args:
            **kwargs: Metadata fields to set
        """
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)
            else:
                self._metadata.custom[key] = value

    def start(self, filepath: Union[str, Path]) -> None:
        """
        Start recording to WAV file.

        Args:
            filepath: Output WAV file path
        """
        if self._recording:
            raise RuntimeError("Already recording")

        self._filepath = Path(filepath)
        self._samples_written = 0
        self._peak_level = 0.0
        self._start_time = datetime.utcnow()
        self._metadata.start_time = self._start_time.isoformat() + "Z"

        # Open WAV file
        self._wav_file = wave.open(str(self._filepath), "wb")

        # WAV parameters
        sample_width = self._get_sample_width()
        # WAV float32 stores as 16-bit internally for compatibility
        if self._sample_format == AudioSampleFormat.FLOAT32:
            sample_width = 2

        self._wav_file.setnchannels(self._channels)
        self._wav_file.setsampwidth(sample_width)
        self._wav_file.setframerate(self._sample_rate)

        self._recording = True

    def write(self, samples: np.ndarray) -> int:
        """
        Write audio samples to recording.

        Args:
            samples: Real audio samples (float, -1.0 to 1.0)

        Returns:
            Number of samples written
        """
        if not self._recording:
            raise RuntimeError("Not recording")

        # Convert and write
        data = self._convert_samples(samples)
        self._wav_file.writeframes(data)

        n_samples = len(samples.flatten())
        self._samples_written += n_samples

        return n_samples

    def stop(self) -> AudioMetadata:
        """
        Stop recording and finalize WAV file.

        Returns:
            Recording metadata
        """
        if not self._recording:
            raise RuntimeError("Not recording")

        # Update metadata
        self._metadata.num_samples = self._samples_written
        self._metadata.duration_seconds = self.duration_seconds

        # Close file
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

        self._recording = False
        return self._metadata

    def get_metadata(self) -> AudioMetadata:
        """Get current metadata."""
        return self._metadata

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._recording:
            self.stop()
        return False


class AudioPlayer:
    """
    Audio playback from WAV files.

    Reads recorded WAV audio files and provides samples for
    processing or playback.

    Features:
    - WAV file support (8-bit, 16-bit, 24-bit)
    - Mono and stereo support
    - Streaming playback
    - Seek and loop support

    Example:
        player = AudioPlayer("audio.wav")
        while not player.eof:
            samples = player.read(1024)
            play(samples)
        player.close()
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize audio player.

        Args:
            filepath: Path to WAV file
        """
        self._filepath = Path(filepath)

        # Playback state
        self._wav_file: Optional[wave.Wave_read] = None
        self._position = 0
        self._eof = False

        # Audio parameters
        self._sample_rate = 0
        self._channels = 0
        self._sample_width = 0
        self._total_samples = 0

        # Metadata
        self._metadata: Optional[AudioMetadata] = None

        # Open file
        self._open()

    def _open(self) -> None:
        """Open the WAV file."""
        self._wav_file = wave.open(str(self._filepath), "rb")

        # Get parameters
        self._channels = self._wav_file.getnchannels()
        self._sample_width = self._wav_file.getsampwidth()
        self._sample_rate = self._wav_file.getframerate()
        self._total_samples = self._wav_file.getnframes()

        # Determine format
        if self._sample_width == 1:
            sample_format = AudioSampleFormat.UINT8
        elif self._sample_width == 2:
            sample_format = AudioSampleFormat.INT16
        elif self._sample_width == 3:
            sample_format = AudioSampleFormat.INT24
        else:
            sample_format = AudioSampleFormat.INT16

        self._metadata = AudioMetadata(
            sample_rate=self._sample_rate,
            sample_format=sample_format,
            channels=self._channels,
            num_samples=self._total_samples,
            duration_seconds=self._total_samples / self._sample_rate,
        )

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self._channels

    @property
    def total_samples(self) -> int:
        """Get total number of samples."""
        return self._total_samples

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        return self._total_samples / self._sample_rate if self._sample_rate > 0 else 0.0

    @property
    def position(self) -> int:
        """Get current sample position."""
        return self._position

    @property
    def position_seconds(self) -> float:
        """Get current position in seconds."""
        return self._position / self._sample_rate if self._sample_rate > 0 else 0.0

    @property
    def eof(self) -> bool:
        """Check if at end of file."""
        return self._eof

    @property
    def metadata(self) -> Optional[AudioMetadata]:
        """Get recording metadata."""
        return self._metadata

    def read(self, num_samples: int) -> np.ndarray:
        """
        Read audio samples from file.

        Args:
            num_samples: Number of samples to read

        Returns:
            Float samples array (-1.0 to 1.0)
        """
        if self._wav_file is None:
            return np.array([], dtype=np.float32)

        frames = self._wav_file.readframes(num_samples)

        if len(frames) == 0:
            self._eof = True
            return np.array([], dtype=np.float32)

        # Convert bytes to samples
        if self._sample_width == 1:
            # Unsigned 8-bit
            samples = np.frombuffer(frames, dtype=np.uint8)
            samples = (samples.astype(np.float32) - 128) / 128.0

        elif self._sample_width == 2:
            # Signed 16-bit
            samples = np.frombuffer(frames, dtype=np.int16)
            samples = samples.astype(np.float32) / 32767.0

        elif self._sample_width == 3:
            # Signed 24-bit (unpack manually)
            n_frames = len(frames) // (3 * self._channels)
            samples = np.zeros(n_frames * self._channels, dtype=np.float32)
            for i in range(len(samples)):
                offset = i * 3
                if offset + 3 <= len(frames):
                    b = frames[offset : offset + 3]
                    val = struct.unpack(
                        "<i", b + (b"\xff" if b[2] & 0x80 else b"\x00")
                    )[0]
                    samples[i] = val / 8388607.0
        else:
            # Default to 16-bit
            samples = np.frombuffer(frames, dtype=np.int16)
            samples = samples.astype(np.float32) / 32767.0

        # Handle multi-channel: return as (n_samples, n_channels) or flatten for mono
        if self._channels > 1:
            n_frames_read = len(samples) // self._channels
            samples = samples.reshape(n_frames_read, self._channels)

        self._position += len(samples) if self._channels == 1 else samples.shape[0]

        return samples

    def read_all(self) -> np.ndarray:
        """
        Read entire file into memory.

        Returns:
            All samples as float array
        """
        self.seek(0)
        return self.read(self._total_samples)

    def seek(self, position: int) -> None:
        """
        Seek to sample position.

        Args:
            position: Sample position
        """
        if self._wav_file is None:
            return

        position = max(0, min(position, self._total_samples - 1))
        self._wav_file.setpos(position)
        self._position = position
        self._eof = False

    def seek_seconds(self, seconds: float) -> None:
        """
        Seek to time position.

        Args:
            seconds: Position in seconds
        """
        position = int(seconds * self._sample_rate)
        self.seek(position)

    def rewind(self) -> None:
        """Rewind to beginning."""
        self.seek(0)

    def close(self) -> None:
        """Close the file."""
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_audio_file(
    filepath: Union[str, Path], max_samples: Optional[int] = None
) -> Tuple[np.ndarray, AudioMetadata]:
    """
    Load entire audio file into memory.

    Args:
        filepath: Path to audio file
        max_samples: Maximum samples to load

    Returns:
        Tuple of (samples, metadata)
    """
    with AudioPlayer(filepath) as player:
        if max_samples is None:
            samples = player.read_all()
        else:
            samples = player.read(max_samples)
        metadata = player.metadata

    return samples, metadata


def save_audio_file(
    filepath: Union[str, Path],
    samples: np.ndarray,
    sample_rate: int = 44100,
    sample_format: AudioSampleFormat = AudioSampleFormat.INT16,
) -> AudioMetadata:
    """
    Save samples to audio WAV file.

    Args:
        filepath: Output file path
        samples: Audio samples (float, -1.0 to 1.0)
        sample_rate: Sample rate in Hz
        sample_format: Sample format

    Returns:
        Recording metadata
    """
    recorder = AudioRecorder(
        sample_rate=sample_rate, sample_format=sample_format, channels=1
    )

    recorder.start(filepath)
    recorder.write(samples)
    return recorder.stop()


class PlaybackMode(Enum):
    """Playback modes."""

    ONCE = "once"  # Play once and stop
    LOOP = "loop"  # Loop continuously
    PING_PONG = "ping_pong"  # Forward then reverse


class PlaybackState(Enum):
    """Playback state."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass
class PlaybackConfig:
    """Playback configuration."""

    mode: PlaybackMode = PlaybackMode.ONCE
    speed: float = 1.0  # Playback speed multiplier
    start_position: int = 0  # Start position in samples
    end_position: int = -1  # End position (-1 = end of file)
    fade_in_samples: int = 0  # Fade in duration
    fade_out_samples: int = 0  # Fade out duration
    gain: float = 1.0  # Output gain
    loop_count: int = -1  # Number of loops (-1 = infinite)


class SignalPlayback:
    """
    Advanced signal playback with looping and speed control.

    Provides sophisticated playback capabilities for recorded I/Q signals,
    suitable for signal replay, testing, and transmission preparation.

    Features:
    - Multiple playback modes: once, loop, ping-pong
    - Variable speed playback with resampling
    - Fade in/out for smooth transitions
    - Loop points and loop count control
    - Pause/resume support
    - Real-time gain adjustment
    - Streaming output for large files

    Example:
        playback = SignalPlayback("recording.sigmf-data")
        playback.set_mode(PlaybackMode.LOOP)
        while playback.state == PlaybackState.PLAYING:
            samples = playback.get_samples(1024)
            transmit(samples)
    """

    def __init__(
        self,
        source: Union[str, Path, np.ndarray],
        sample_rate: Optional[float] = None,
        config: Optional[PlaybackConfig] = None,
    ):
        """
        Initialize signal playback.

        Args:
            source: File path or numpy array of samples
            sample_rate: Sample rate (required if source is array)
            config: Playback configuration
        """
        self._config = config or PlaybackConfig()
        self._state = PlaybackState.STOPPED

        # Load source
        if isinstance(source, (str, Path)):
            self._player = IQPlayer(source)
            self._samples = self._player.read(self._player.total_samples)
            self._sample_rate = self._player.sample_rate
            self._player.close()
        else:
            self._samples = np.asarray(source, dtype=np.complex64)
            self._sample_rate = sample_rate or 1.0
            self._player = None

        # Playback state
        self._position = self._config.start_position
        self._direction = 1  # 1 = forward, -1 = reverse
        self._loop_counter = 0
        self._total_samples_played = 0

        # End position
        if self._config.end_position < 0:
            self._end_position = len(self._samples)
        else:
            self._end_position = min(self._config.end_position, len(self._samples))

        # Resampling for speed control
        self._resample_buffer = np.array([], dtype=np.complex64)
        self._fractional_position = 0.0

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def output_sample_rate(self) -> float:
        """Get effective output sample rate (after speed adjustment)."""
        return self._sample_rate * self._config.speed

    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state

    @property
    def position(self) -> int:
        """Get current position in samples."""
        return int(self._position)

    @property
    def position_seconds(self) -> float:
        """Get current position in seconds."""
        return self._position / self._sample_rate

    @property
    def duration_samples(self) -> int:
        """Get total duration in samples."""
        return self._end_position - self._config.start_position

    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        return self.duration_samples / self._sample_rate

    @property
    def progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        total = self._end_position - self._config.start_position
        if total <= 0:
            return 0.0
        current = self._position - self._config.start_position
        return current / total

    @property
    def loop_count(self) -> int:
        """Get number of loops completed."""
        return self._loop_counter

    @property
    def total_samples_played(self) -> int:
        """Get total samples played (including loops)."""
        return self._total_samples_played

    def set_config(self, config: PlaybackConfig) -> None:
        """Update playback configuration."""
        self._config = config
        if config.end_position < 0:
            self._end_position = len(self._samples)
        else:
            self._end_position = min(config.end_position, len(self._samples))

    def set_mode(self, mode: PlaybackMode) -> None:
        """Set playback mode."""
        self._config.mode = mode

    def set_speed(self, speed: float) -> None:
        """Set playback speed (0.5 to 2.0 typical)."""
        self._config.speed = max(0.1, min(10.0, speed))

    def set_gain(self, gain: float) -> None:
        """Set output gain."""
        self._config.gain = gain

    def set_gain_db(self, gain_db: float) -> None:
        """Set output gain in dB."""
        self._config.gain = 10 ** (gain_db / 20)

    def play(self) -> None:
        """Start or resume playback."""
        if self._state == PlaybackState.FINISHED:
            self._position = self._config.start_position
            self._loop_counter = 0
        self._state = PlaybackState.PLAYING

    def pause(self) -> None:
        """Pause playback."""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        """Stop playback and reset position."""
        self._state = PlaybackState.STOPPED
        self._position = self._config.start_position
        self._direction = 1
        self._loop_counter = 0

    def seek(self, position: int) -> None:
        """Seek to sample position."""
        self._position = max(
            self._config.start_position, min(position, self._end_position - 1)
        )

    def seek_seconds(self, seconds: float) -> None:
        """Seek to time position."""
        position = int(seconds * self._sample_rate)
        self.seek(position)

    def _apply_fade(self, samples: np.ndarray, start_pos: int) -> np.ndarray:
        """Apply fade in/out to samples."""
        n = len(samples)
        output = samples.copy()

        # Fade in
        if self._config.fade_in_samples > 0:
            fade_in_end = self._config.start_position + self._config.fade_in_samples
            for i in range(n):
                pos = start_pos + i
                if pos < fade_in_end:
                    fade = (
                        pos - self._config.start_position
                    ) / self._config.fade_in_samples
                    output[i] *= fade

        # Fade out
        if self._config.fade_out_samples > 0:
            fade_out_start = self._end_position - self._config.fade_out_samples
            for i in range(n):
                pos = start_pos + i
                if pos >= fade_out_start:
                    fade = (self._end_position - pos) / self._config.fade_out_samples
                    output[i] *= max(0, fade)

        return output

    def _resample_chunk(self, samples: np.ndarray) -> np.ndarray:
        """Resample chunk for speed adjustment."""
        if abs(self._config.speed - 1.0) < 0.001:
            return samples

        # Simple linear interpolation resampling
        n_in = len(samples)
        n_out = int(n_in / self._config.speed)

        if n_out <= 0:
            return np.array([], dtype=np.complex64)

        indices = np.linspace(0, n_in - 1, n_out)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.minimum(indices_floor + 1, n_in - 1)
        frac = indices - indices_floor

        resampled = (1 - frac) * samples[indices_floor] + frac * samples[indices_ceil]
        return resampled.astype(np.complex64)

    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        Get next batch of samples for playback.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Complex samples array (may be shorter at end)
        """
        if self._state != PlaybackState.PLAYING:
            return np.zeros(num_samples, dtype=np.complex64)

        # Calculate how many source samples we need
        source_samples_needed = int(num_samples * self._config.speed) + 2

        output = []
        samples_remaining = num_samples

        while samples_remaining > 0 and self._state == PlaybackState.PLAYING:
            # Get chunk from source
            if self._direction > 0:
                # Forward playback
                end_idx = min(
                    int(self._position) + source_samples_needed, self._end_position
                )
                chunk = self._samples[int(self._position) : end_idx]
            else:
                # Reverse playback
                start_idx = max(
                    int(self._position) - source_samples_needed,
                    self._config.start_position,
                )
                chunk = self._samples[start_idx : int(self._position)][::-1]

            if len(chunk) == 0:
                # Handle end of segment
                if self._config.mode == PlaybackMode.ONCE:
                    self._state = PlaybackState.FINISHED
                    break

                elif self._config.mode == PlaybackMode.LOOP:
                    self._loop_counter += 1
                    if (
                        self._config.loop_count > 0
                        and self._loop_counter >= self._config.loop_count
                    ):
                        self._state = PlaybackState.FINISHED
                        break
                    self._position = self._config.start_position
                    continue

                elif self._config.mode == PlaybackMode.PING_PONG:
                    self._direction *= -1
                    if self._direction > 0:
                        self._loop_counter += 1
                        if (
                            self._config.loop_count > 0
                            and self._loop_counter >= self._config.loop_count
                        ):
                            self._state = PlaybackState.FINISHED
                            break
                        self._position = self._config.start_position
                    else:
                        self._position = self._end_position - 1
                    continue

            # Apply fade
            chunk = self._apply_fade(chunk, int(self._position))

            # Resample for speed
            resampled = self._resample_chunk(chunk)

            # Take what we need
            take = min(len(resampled), samples_remaining)
            output.append(resampled[:take])
            samples_remaining -= take

            # Update position
            self._position += self._direction * len(chunk)
            self._total_samples_played += take

        if len(output) == 0:
            return np.zeros(num_samples, dtype=np.complex64)

        result = np.concatenate(output)

        # Apply gain
        result *= self._config.gain

        # Pad if needed
        if len(result) < num_samples:
            result = np.pad(result, (0, num_samples - len(result)))

        return result[:num_samples].astype(np.complex64)

    def get_all_samples(self) -> np.ndarray:
        """Get all samples at once (for small files)."""
        start = self._config.start_position
        end = self._end_position
        samples = self._samples[start:end].copy()

        # Apply fade
        samples = self._apply_fade(samples, start)

        # Resample
        samples = self._resample_chunk(samples)

        # Apply gain
        samples *= self._config.gain

        return samples.astype(np.complex64)

    def get_status(self) -> dict:
        """Get playback status."""
        return {
            "state": self._state.value,
            "mode": self._config.mode.value,
            "position": self._position,
            "position_seconds": self.position_seconds,
            "progress": self.progress,
            "duration_seconds": self.duration_seconds,
            "loop_count": self._loop_counter,
            "speed": self._config.speed,
            "gain": self._config.gain,
            "sample_rate": self._sample_rate,
        }


class TransmitBuffer:
    """
    Buffer for preparing signals for transmission.

    Manages signal buffering and flow control for transmit operations,
    ensuring smooth continuous transmission without underruns.

    Features:
    - Ring buffer for continuous transmission
    - Underrun detection and handling
    - Pre-buffering before transmission starts
    - Multiple signal source support
    - Gain and power control
    - Transmit burst support

    Example:
        buffer = TransmitBuffer(sample_rate=2.4e6, buffer_seconds=1.0)
        buffer.add_samples(modulated_signal)
        while transmitting:
            samples = buffer.get_transmit_samples(1024)
            hackrf.transmit(samples)
    """

    def __init__(
        self,
        sample_rate: float,
        buffer_seconds: float = 1.0,
        prebuffer_seconds: float = 0.1,
    ):
        """
        Initialize transmit buffer.

        Args:
            sample_rate: Transmit sample rate in Hz
            buffer_seconds: Total buffer size in seconds
            prebuffer_seconds: Minimum buffer before transmission
        """
        self._sample_rate = sample_rate
        self._buffer_size = int(buffer_seconds * sample_rate)
        self._prebuffer_size = int(prebuffer_seconds * sample_rate)

        # Ring buffer
        self._buffer = np.zeros(self._buffer_size, dtype=np.complex64)
        self._write_pos = 0
        self._read_pos = 0
        self._samples_available = 0

        # State
        self._transmitting = False
        self._underrun_count = 0
        self._total_transmitted = 0

        # Gain control
        self._gain = 1.0
        self._peak_limit = 1.0  # Maximum amplitude

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def buffer_size(self) -> int:
        """Get buffer size in samples."""
        return self._buffer_size

    @property
    def samples_available(self) -> int:
        """Get number of samples available for transmission."""
        return self._samples_available

    @property
    def buffer_fullness(self) -> float:
        """Get buffer fullness (0.0 to 1.0)."""
        return self._samples_available / self._buffer_size

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough samples to start transmission."""
        return self._samples_available >= self._prebuffer_size

    @property
    def is_transmitting(self) -> bool:
        """Check if currently transmitting."""
        return self._transmitting

    @property
    def underrun_count(self) -> int:
        """Get number of buffer underruns."""
        return self._underrun_count

    @property
    def total_transmitted(self) -> int:
        """Get total samples transmitted."""
        return self._total_transmitted

    def set_gain(self, gain: float) -> None:
        """Set output gain."""
        self._gain = gain

    def set_gain_db(self, gain_db: float) -> None:
        """Set output gain in dB."""
        self._gain = 10 ** (gain_db / 20)

    def set_peak_limit(self, limit: float) -> None:
        """Set peak amplitude limit."""
        self._peak_limit = limit

    def add_samples(self, samples: np.ndarray) -> int:
        """
        Add samples to transmit buffer.

        Args:
            samples: Complex samples to add

        Returns:
            Number of samples actually added
        """
        samples = np.asarray(samples, dtype=np.complex64)
        n = len(samples)

        # Check available space
        space_available = self._buffer_size - self._samples_available

        if n > space_available:
            # Buffer overflow - drop oldest samples or clip
            n = space_available
            samples = samples[:n]

        if n == 0:
            return 0

        # Write to ring buffer
        for i in range(n):
            self._buffer[self._write_pos] = samples[i]
            self._write_pos = (self._write_pos + 1) % self._buffer_size

        self._samples_available += n
        return n

    def add_from_playback(self, playback: SignalPlayback, num_samples: int) -> int:
        """
        Add samples from a playback source.

        Args:
            playback: SignalPlayback instance
            num_samples: Number of samples to fetch and add

        Returns:
            Number of samples added
        """
        samples = playback.get_samples(num_samples)
        return self.add_samples(samples)

    def get_transmit_samples(self, num_samples: int) -> np.ndarray:
        """
        Get samples for transmission.

        Args:
            num_samples: Number of samples needed

        Returns:
            Complex samples ready for transmission
        """
        if not self._transmitting:
            if self.is_ready:
                self._transmitting = True
            else:
                # Not ready - return zeros (or could return None)
                return np.zeros(num_samples, dtype=np.complex64)

        # Check for underrun
        if self._samples_available < num_samples:
            self._underrun_count += 1
            # Return what we have plus zeros
            available = self._samples_available
        else:
            available = num_samples

        # Read from ring buffer
        output = np.zeros(num_samples, dtype=np.complex64)
        for i in range(available):
            output[i] = self._buffer[self._read_pos]
            self._read_pos = (self._read_pos + 1) % self._buffer_size

        self._samples_available -= available

        # Apply gain
        output *= self._gain

        # Apply peak limiting (soft clipping)
        magnitudes = np.abs(output)
        over_limit = magnitudes > self._peak_limit
        if np.any(over_limit):
            phases = np.angle(output)
            magnitudes[over_limit] = self._peak_limit
            output = magnitudes * np.exp(1j * phases)

        self._total_transmitted += available

        return output

    def start_transmission(self) -> bool:
        """
        Start transmission (waits for buffer ready).

        Returns:
            True if started, False if buffer not ready
        """
        if self.is_ready:
            self._transmitting = True
            return True
        return False

    def stop_transmission(self) -> None:
        """Stop transmission."""
        self._transmitting = False

    def flush(self) -> None:
        """Clear the buffer."""
        self._buffer.fill(0)
        self._write_pos = 0
        self._read_pos = 0
        self._samples_available = 0

    def get_status(self) -> dict:
        """Get buffer status."""
        return {
            "transmitting": self._transmitting,
            "samples_available": self._samples_available,
            "buffer_fullness": self.buffer_fullness,
            "is_ready": self.is_ready,
            "underrun_count": self._underrun_count,
            "total_transmitted": self._total_transmitted,
            "gain": self._gain,
            "peak_limit": self._peak_limit,
        }


class PlaybackScheduler:
    """
    Schedule signal playback at specific times or intervals.

    Manages timed playback of signals for automated testing,
    beacon transmission, or scheduled broadcasts.

    Features:
    - Schedule playback at absolute or relative times
    - Repeat schedules (hourly, daily, custom intervals)
    - Multiple playback sources
    - Priority-based scheduling
    - Event callbacks

    Example:
        scheduler = PlaybackScheduler()
        scheduler.add_scheduled_playback(
            "beacon",
            playback=SignalPlayback("beacon.iq"),
            interval_seconds=60
        )
        scheduler.start()
    """

    def __init__(self):
        """Initialize playback scheduler."""
        self._schedules: Dict[str, dict] = {}
        self._running = False
        self._current_playback: Optional[str] = None

    def add_scheduled_playback(
        self,
        name: str,
        playback: SignalPlayback,
        start_time: Optional[datetime] = None,
        interval_seconds: Optional[float] = None,
        repeat_count: int = -1,
        priority: int = 0,
    ) -> None:
        """
        Add a scheduled playback.

        Args:
            name: Unique identifier for this schedule
            playback: SignalPlayback instance
            start_time: When to start (None = immediately when run)
            interval_seconds: Repeat interval (None = once only)
            repeat_count: Number of repeats (-1 = infinite)
            priority: Higher priority interrupts lower
        """
        self._schedules[name] = {
            "playback": playback,
            "start_time": start_time,
            "interval": interval_seconds,
            "repeat_count": repeat_count,
            "repeats_done": 0,
            "priority": priority,
            "last_run": None,
            "next_run": start_time or datetime.utcnow(),
            "enabled": True,
        }

    def remove_scheduled_playback(self, name: str) -> bool:
        """Remove a scheduled playback."""
        if name in self._schedules:
            del self._schedules[name]
            return True
        return False

    def enable_schedule(self, name: str) -> None:
        """Enable a schedule."""
        if name in self._schedules:
            self._schedules[name]["enabled"] = True

    def disable_schedule(self, name: str) -> None:
        """Disable a schedule."""
        if name in self._schedules:
            self._schedules[name]["enabled"] = False

    def get_next_scheduled(self) -> Optional[Tuple[str, datetime]]:
        """Get the next scheduled playback."""
        next_name = None
        next_time = None

        for name, schedule in self._schedules.items():
            if not schedule["enabled"]:
                continue

            if schedule["repeat_count"] >= 0:
                if schedule["repeats_done"] >= schedule["repeat_count"]:
                    continue

            run_time = schedule["next_run"]
            if run_time and (next_time is None or run_time < next_time):
                next_name = name
                next_time = run_time

        if next_name:
            return (next_name, next_time)
        return None

    def check_and_get_playback(self) -> Optional[Tuple[str, SignalPlayback]]:
        """
        Check if any playback is due and return it.

        Returns:
            Tuple of (name, playback) if due, None otherwise
        """
        now = datetime.utcnow()
        best_candidate = None
        best_priority = -1

        for name, schedule in self._schedules.items():
            if not schedule["enabled"]:
                continue

            # Check repeat limit
            if schedule["repeat_count"] >= 0:
                if schedule["repeats_done"] >= schedule["repeat_count"]:
                    continue

            # Check if due
            next_run = schedule["next_run"]
            if next_run and next_run <= now:
                if schedule["priority"] > best_priority:
                    best_candidate = name
                    best_priority = schedule["priority"]

        if best_candidate:
            schedule = self._schedules[best_candidate]

            # Update schedule
            schedule["last_run"] = now
            schedule["repeats_done"] += 1

            if schedule["interval"]:
                from datetime import timedelta

                schedule["next_run"] = now + timedelta(seconds=schedule["interval"])
            else:
                schedule["next_run"] = None

            # Reset playback
            playback = schedule["playback"]
            playback.stop()
            playback.play()

            self._current_playback = best_candidate
            return (best_candidate, playback)

        return None

    def get_schedule_status(self) -> Dict[str, dict]:
        """Get status of all schedules."""
        status = {}
        for name, schedule in self._schedules.items():
            status[name] = {
                "enabled": schedule["enabled"],
                "next_run": (
                    schedule["next_run"].isoformat() if schedule["next_run"] else None
                ),
                "last_run": (
                    schedule["last_run"].isoformat() if schedule["last_run"] else None
                ),
                "repeats_done": schedule["repeats_done"],
                "repeat_count": schedule["repeat_count"],
                "priority": schedule["priority"],
            }
        return status

    def reset_all(self) -> None:
        """Reset all schedules to initial state."""
        for schedule in self._schedules.values():
            schedule["repeats_done"] = 0
            schedule["last_run"] = None
            if schedule["start_time"]:
                schedule["next_run"] = schedule["start_time"]
            else:
                schedule["next_run"] = datetime.utcnow()


# =============================================================================
# File Format Utilities
# =============================================================================


@dataclass
class FormatInfo:
    """Detailed information about a file format."""

    file_format: FileFormat
    sample_format: SampleFormat
    sample_rate: float = 0.0
    center_frequency: float = 0.0
    num_samples: int = 0
    num_channels: int = 2  # 2 for I/Q
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    bits_per_sample: int = 0
    is_valid: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class FormatDetector:
    """
    Auto-detect SDR file formats from file content and extension.

    Supports detection of:
    - Raw I/Q files (.raw, .iq, .cu8, .cs8, .cs16, .cf32, .cf64)
    - WAV files (.wav)
    - SigMF files (.sigmf-data, .sigmf-meta)

    Example:
        detector = FormatDetector()
        info = detector.detect("recording.sigmf-data")
        print(f"Format: {info.file_format}, Sample rate: {info.sample_rate}")
    """

    # Magic bytes for format detection
    WAV_MAGIC = b"RIFF"
    SIGMF_META_KEYS = ["global", "captures", "annotations"]

    # Extension to format mapping
    EXTENSION_MAP = {
        ".wav": (FileFormat.WAV, None),
        ".cu8": (FileFormat.RAW, SampleFormat.UINT8),
        ".cs8": (FileFormat.RAW, SampleFormat.INT8),
        ".cs16": (FileFormat.RAW, SampleFormat.INT16),
        ".cf32": (FileFormat.RAW, SampleFormat.FLOAT32),
        ".cf64": (FileFormat.RAW, SampleFormat.FLOAT64),
        ".raw": (FileFormat.RAW, SampleFormat.FLOAT32),
        ".iq": (FileFormat.RAW, SampleFormat.FLOAT32),
        ".bin": (FileFormat.RAW, SampleFormat.FLOAT32),
        ".sigmf-data": (FileFormat.SIGMF, None),
        ".sigmf-meta": (FileFormat.SIGMF, None),
    }

    # Bytes per sample for each format
    BYTES_PER_SAMPLE = {
        SampleFormat.UINT8: 1,
        SampleFormat.INT8: 1,
        SampleFormat.INT16: 2,
        SampleFormat.FLOAT32: 4,
        SampleFormat.FLOAT64: 8,
    }

    def detect(self, filepath: Union[str, Path]) -> FormatInfo:
        """
        Detect file format and extract information.

        Args:
            filepath: Path to the file

        Returns:
            FormatInfo with detected parameters
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return FormatInfo(
                file_format=FileFormat.RAW,
                sample_format=SampleFormat.FLOAT32,
                is_valid=False,
                error_message=f"File not found: {filepath}",
            )

        # Get file size
        file_size = filepath.stat().st_size

        # Try to detect by extension first
        suffix = filepath.suffix.lower()
        file_format, sample_format = self.EXTENSION_MAP.get(
            suffix, (FileFormat.RAW, SampleFormat.FLOAT32)
        )

        # Detect by magic bytes / content
        try:
            if suffix == ".wav" or self._is_wav_file(filepath):
                return self._detect_wav(filepath)

            elif suffix in (".sigmf-data", ".sigmf-meta") or self._has_sigmf_meta(
                filepath
            ):
                return self._detect_sigmf(filepath)

            else:
                return self._detect_raw(filepath, sample_format)

        except Exception as e:
            return FormatInfo(
                file_format=file_format,
                sample_format=sample_format or SampleFormat.FLOAT32,
                file_size_bytes=file_size,
                is_valid=False,
                error_message=str(e),
            )

    def _is_wav_file(self, filepath: Path) -> bool:
        """Check if file is a WAV file by magic bytes."""
        try:
            with open(filepath, "rb") as f:
                magic = f.read(4)
                return magic == self.WAV_MAGIC
        except Exception:
            return False

    def _has_sigmf_meta(self, filepath: Path) -> bool:
        """Check if SigMF metadata file exists."""
        meta_path = filepath.with_suffix(".sigmf-meta")
        return meta_path.exists()

    def _detect_wav(self, filepath: Path) -> FormatInfo:
        """Detect WAV file format."""
        try:
            with wave.open(str(filepath), "rb") as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()

                # Map sample width to format
                if sample_width == 1:
                    sample_format = SampleFormat.UINT8
                elif sample_width == 2:
                    sample_format = SampleFormat.INT16
                else:
                    sample_format = SampleFormat.INT16

                file_size = filepath.stat().st_size
                duration = n_frames / sample_rate if sample_rate > 0 else 0

                return FormatInfo(
                    file_format=FileFormat.WAV,
                    sample_format=sample_format,
                    sample_rate=float(sample_rate),
                    num_samples=n_frames,
                    num_channels=n_channels,
                    file_size_bytes=file_size,
                    duration_seconds=duration,
                    bits_per_sample=sample_width * 8,
                    is_valid=True,
                    metadata={
                        "wav_channels": n_channels,
                        "wav_sample_width": sample_width,
                    },
                )
        except Exception as e:
            return FormatInfo(
                file_format=FileFormat.WAV,
                sample_format=SampleFormat.INT16,
                is_valid=False,
                error_message=str(e),
            )

    def _detect_sigmf(self, filepath: Path) -> FormatInfo:
        """Detect SigMF file format."""
        # Find metadata file
        if filepath.suffix == ".sigmf-meta":
            meta_path = filepath
            data_path = filepath.with_suffix(".sigmf-data")
        else:
            data_path = filepath.with_suffix(".sigmf-data")
            meta_path = filepath.with_suffix(".sigmf-meta")

        if not data_path.exists():
            data_path = filepath

        try:
            # Read metadata
            sigmf_meta = {}
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    sigmf_meta = json.load(f)

            global_meta = sigmf_meta.get("global", {})
            captures = sigmf_meta.get("captures", [{}])

            # Parse datatype
            datatype = global_meta.get("core:datatype", "cf32_le")
            format_map = {
                "cu8": SampleFormat.UINT8,
                "ci8": SampleFormat.INT8,
                "ci16_le": SampleFormat.INT16,
                "ci16_be": SampleFormat.INT16,
                "cf32_le": SampleFormat.FLOAT32,
                "cf32_be": SampleFormat.FLOAT32,
                "cf64_le": SampleFormat.FLOAT64,
                "cf64_be": SampleFormat.FLOAT64,
            }
            sample_format = format_map.get(datatype, SampleFormat.FLOAT32)

            sample_rate = global_meta.get("core:sample_rate", 0.0)
            center_freq = captures[0].get("core:frequency", 0.0) if captures else 0.0

            # Calculate samples from file size
            file_size = data_path.stat().st_size if data_path.exists() else 0
            bytes_per_sample = self.BYTES_PER_SAMPLE.get(sample_format, 4) * 2
            num_samples = file_size // bytes_per_sample
            duration = num_samples / sample_rate if sample_rate > 0 else 0

            return FormatInfo(
                file_format=FileFormat.SIGMF,
                sample_format=sample_format,
                sample_rate=sample_rate,
                center_frequency=center_freq,
                num_samples=num_samples,
                num_channels=2,
                file_size_bytes=file_size,
                duration_seconds=duration,
                bits_per_sample=self.BYTES_PER_SAMPLE.get(sample_format, 4) * 8,
                is_valid=True,
                metadata={
                    "sigmf_version": global_meta.get("core:version", ""),
                    "sigmf_description": global_meta.get("core:description", ""),
                    "sigmf_author": global_meta.get("core:author", ""),
                    "sigmf_hardware": global_meta.get("core:hw", ""),
                    "sigmf_datatype": datatype,
                },
            )
        except Exception as e:
            return FormatInfo(
                file_format=FileFormat.SIGMF,
                sample_format=SampleFormat.FLOAT32,
                is_valid=False,
                error_message=str(e),
            )

    def _detect_raw(
        self, filepath: Path, sample_format: Optional[SampleFormat] = None
    ) -> FormatInfo:
        """Detect raw I/Q file format."""
        file_size = filepath.stat().st_size
        sample_format = sample_format or SampleFormat.FLOAT32

        bytes_per_sample = self.BYTES_PER_SAMPLE.get(sample_format, 4) * 2
        num_samples = file_size // bytes_per_sample

        return FormatInfo(
            file_format=FileFormat.RAW,
            sample_format=sample_format,
            num_samples=num_samples,
            num_channels=2,
            file_size_bytes=file_size,
            bits_per_sample=self.BYTES_PER_SAMPLE.get(sample_format, 4) * 8,
            is_valid=True,
            metadata={
                "detected_from": "extension",
            },
        )

    def detect_sample_format_from_data(
        self, filepath: Union[str, Path], sample_count: int = 1000
    ) -> SampleFormat:
        """
        Attempt to detect sample format by analyzing data values.

        Args:
            filepath: Path to raw file
            sample_count: Number of samples to analyze

        Returns:
            Most likely SampleFormat
        """
        filepath = Path(filepath)
        file_size = filepath.stat().st_size

        # Read sample data
        with open(filepath, "rb") as f:
            data = f.read(min(file_size, sample_count * 16))

        # Try different interpretations
        scores = {}

        # UINT8: values 0-255, typically centered around 127-128
        try:
            samples = np.frombuffer(data, dtype=np.uint8)
            if len(samples) > 10:
                mean_val = np.mean(samples)
                if 100 < mean_val < 156:  # Near center (127.5)
                    scores[SampleFormat.UINT8] = 1.0 - abs(mean_val - 127.5) / 127.5
        except Exception:
            pass

        # INT16: values typically in reasonable range
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            if len(samples) > 10:
                max_abs = np.max(np.abs(samples))
                if max_abs < 32768:
                    scores[SampleFormat.INT16] = 0.8 if max_abs > 100 else 0.3
        except Exception:
            pass

        # FLOAT32: check for valid float values
        try:
            samples = np.frombuffer(data, dtype=np.float32)
            if len(samples) > 10:
                if not np.any(np.isnan(samples)) and not np.any(np.isinf(samples)):
                    max_abs = np.max(np.abs(samples))
                    if max_abs < 10.0:  # Typically normalized
                        scores[SampleFormat.FLOAT32] = 0.9
                    elif max_abs < 1000:
                        scores[SampleFormat.FLOAT32] = 0.5
        except Exception:
            pass

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        return SampleFormat.FLOAT32


class FormatConverter:
    """
    Convert between SDR file formats.

    Supports conversion between:
    - Raw I/Q (various sample formats)
    - WAV (stereo I/Q)
    - SigMF (with metadata)

    Example:
        converter = FormatConverter()
        converter.convert(
            "input.cu8",
            "output.sigmf-data",
            input_format=SampleFormat.UINT8,
            output_format=SampleFormat.FLOAT32,
            sample_rate=2.4e6
        )
    """

    def __init__(self, chunk_size: int = 65536):
        """
        Initialize converter.

        Args:
            chunk_size: Samples per chunk for streaming conversion
        """
        self._chunk_size = chunk_size
        self._detector = FormatDetector()

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        input_format: Optional[SampleFormat] = None,
        output_format: SampleFormat = SampleFormat.FLOAT32,
        output_file_format: Optional[FileFormat] = None,
        sample_rate: float = 0.0,
        center_frequency: float = 0.0,
        description: str = "",
    ) -> FormatInfo:
        """
        Convert file to different format.

        Args:
            input_path: Input file path
            output_path: Output file path
            input_format: Input sample format (auto-detect if None)
            output_format: Output sample format
            output_file_format: Output file format (detect from extension if None)
            sample_rate: Sample rate (for metadata)
            center_frequency: Center frequency (for metadata)
            description: File description (for metadata)

        Returns:
            FormatInfo for output file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Detect input format
        input_info = self._detector.detect(input_path)
        if not input_info.is_valid:
            raise ValueError(f"Cannot read input file: {input_info.error_message}")

        if input_format:
            input_info.sample_format = input_format
        if sample_rate > 0:
            input_info.sample_rate = sample_rate
        if center_frequency > 0:
            input_info.center_frequency = center_frequency

        # Determine output file format
        if output_file_format is None:
            suffix = output_path.suffix.lower()
            if suffix == ".wav":
                output_file_format = FileFormat.WAV
            elif suffix in (".sigmf-data", ".sigmf-meta"):
                output_file_format = FileFormat.SIGMF
            else:
                output_file_format = FileFormat.RAW

        # Load input samples
        samples, _ = load_iq_file(input_path, input_info.sample_format)

        # Save to output format
        if output_file_format == FileFormat.WAV:
            return self._save_wav(
                output_path, samples, output_format, input_info.sample_rate
            )
        elif output_file_format == FileFormat.SIGMF:
            return self._save_sigmf(
                output_path,
                samples,
                output_format,
                input_info.sample_rate,
                input_info.center_frequency,
                description,
            )
        else:
            return self._save_raw(
                output_path,
                samples,
                output_format,
                input_info.sample_rate,
                input_info.center_frequency,
            )

    def _save_raw(
        self,
        filepath: Path,
        samples: np.ndarray,
        sample_format: SampleFormat,
        sample_rate: float,
        center_frequency: float,
    ) -> FormatInfo:
        """Save to raw I/Q format."""
        save_iq_file(
            filepath,
            samples,
            sample_rate,
            center_frequency,
            sample_format,
            FileFormat.RAW,
        )
        return FormatInfo(
            file_format=FileFormat.RAW,
            sample_format=sample_format,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            num_samples=len(samples),
            file_size_bytes=filepath.stat().st_size,
            duration_seconds=len(samples) / sample_rate if sample_rate > 0 else 0,
            is_valid=True,
        )

    def _save_wav(
        self,
        filepath: Path,
        samples: np.ndarray,
        sample_format: SampleFormat,
        sample_rate: float,
    ) -> FormatInfo:
        """Save to WAV format."""
        save_iq_file(filepath, samples, sample_rate, 0.0, sample_format, FileFormat.WAV)
        return FormatInfo(
            file_format=FileFormat.WAV,
            sample_format=sample_format,
            sample_rate=sample_rate,
            num_samples=len(samples),
            file_size_bytes=filepath.stat().st_size,
            duration_seconds=len(samples) / sample_rate if sample_rate > 0 else 0,
            is_valid=True,
        )

    def _save_sigmf(
        self,
        filepath: Path,
        samples: np.ndarray,
        sample_format: SampleFormat,
        sample_rate: float,
        center_frequency: float,
        description: str,
    ) -> FormatInfo:
        """Save to SigMF format."""
        recorder = IQRecorder(
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            sample_format=sample_format,
            file_format=FileFormat.SIGMF,
        )
        recorder.set_metadata(description=description)
        recorder.start(filepath)
        recorder.write(samples)
        recorder.stop()

        data_path = filepath.with_suffix(".sigmf-data")
        file_size = data_path.stat().st_size if data_path.exists() else 0

        return FormatInfo(
            file_format=FileFormat.SIGMF,
            sample_format=sample_format,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            num_samples=len(samples),
            file_size_bytes=file_size,
            duration_seconds=len(samples) / sample_rate if sample_rate > 0 else 0,
            is_valid=True,
            metadata={"description": description},
        )

    def convert_sample_format(
        self, samples: np.ndarray, target_format: SampleFormat
    ) -> np.ndarray:
        """
        Convert samples to different sample format (in memory).

        Args:
            samples: Input complex samples
            target_format: Target sample format

        Returns:
            Converted samples as interleaved I/Q array
        """
        # Interleave I/Q
        interleaved = np.zeros(len(samples) * 2, dtype=np.float64)
        interleaved[0::2] = samples.real
        interleaved[1::2] = samples.imag

        # Convert to target format
        if target_format == SampleFormat.UINT8:
            scaled = np.clip(interleaved * 127.5 + 127.5, 0, 255)
            return scaled.astype(np.uint8)

        elif target_format == SampleFormat.INT8:
            scaled = np.clip(interleaved * 127, -128, 127)
            return scaled.astype(np.int8)

        elif target_format == SampleFormat.INT16:
            scaled = np.clip(interleaved * 32767, -32768, 32767)
            return scaled.astype(np.int16)

        elif target_format == SampleFormat.FLOAT32:
            return interleaved.astype(np.float32)

        elif target_format == SampleFormat.FLOAT64:
            return interleaved.astype(np.float64)

        return interleaved.astype(np.float32)

    def resample_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_sample_rate: float,
        input_sample_rate: Optional[float] = None,
    ) -> FormatInfo:
        """
        Resample a file to different sample rate.

        Args:
            input_path: Input file
            output_path: Output file
            target_sample_rate: Target sample rate
            input_sample_rate: Input sample rate (auto-detect if None)

        Returns:
            FormatInfo for output file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Load input
        input_info = self._detector.detect(input_path)
        samples, _ = load_iq_file(input_path, input_info.sample_format)

        source_rate = input_sample_rate or input_info.sample_rate
        if source_rate <= 0:
            raise ValueError(
                "Input sample rate not specified and could not be detected"
            )

        # Calculate resampling ratio
        ratio = target_sample_rate / source_rate

        # Simple linear interpolation resampling
        n_output = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, n_output)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.minimum(indices_floor + 1, len(samples) - 1)
        frac = indices - indices_floor

        resampled = (
            (1 - frac) * samples[indices_floor] + frac * samples[indices_ceil]
        ).astype(np.complex64)

        # Save resampled
        output_info = (
            self._detector.detect(output_path) if output_path.exists() else None
        )
        output_format = (
            output_info.sample_format if output_info else input_info.sample_format
        )

        suffix = output_path.suffix.lower()
        if suffix == ".wav":
            file_format = FileFormat.WAV
        elif suffix in (".sigmf-data", ".sigmf-meta"):
            file_format = FileFormat.SIGMF
        else:
            file_format = FileFormat.RAW

        save_iq_file(
            output_path,
            resampled,
            target_sample_rate,
            input_info.center_frequency,
            output_format,
            file_format,
        )

        return FormatInfo(
            file_format=file_format,
            sample_format=output_format,
            sample_rate=target_sample_rate,
            center_frequency=input_info.center_frequency,
            num_samples=len(resampled),
            file_size_bytes=output_path.stat().st_size,
            duration_seconds=len(resampled) / target_sample_rate,
            is_valid=True,
            metadata={"resampled_from": source_rate},
        )


def detect_format(filepath: Union[str, Path]) -> FormatInfo:
    """
    Convenience function to detect file format.

    Args:
        filepath: Path to file

    Returns:
        FormatInfo with detected parameters
    """
    return FormatDetector().detect(filepath)


def convert_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: SampleFormat = SampleFormat.FLOAT32,
    sample_rate: float = 0.0,
    center_frequency: float = 0.0,
) -> FormatInfo:
    """
    Convenience function to convert file format.

    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Output sample format
        sample_rate: Sample rate (for metadata)
        center_frequency: Center frequency (for metadata)

    Returns:
        FormatInfo for output file
    """
    return FormatConverter().convert(
        input_path,
        output_path,
        output_format=output_format,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
    )


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get human-readable file information.

    Args:
        filepath: Path to file

    Returns:
        Dictionary with file information
    """
    info = detect_format(filepath)

    # Format duration nicely
    if info.duration_seconds > 0:
        if info.duration_seconds >= 3600:
            hours = int(info.duration_seconds // 3600)
            minutes = int((info.duration_seconds % 3600) // 60)
            seconds = info.duration_seconds % 60
            duration_str = f"{hours}h {minutes}m {seconds:.1f}s"
        elif info.duration_seconds >= 60:
            minutes = int(info.duration_seconds // 60)
            seconds = info.duration_seconds % 60
            duration_str = f"{minutes}m {seconds:.1f}s"
        else:
            duration_str = f"{info.duration_seconds:.2f}s"
    else:
        duration_str = "unknown"

    # Format file size nicely
    if info.file_size_bytes >= 1024 * 1024 * 1024:
        size_str = f"{info.file_size_bytes / (1024**3):.2f} GB"
    elif info.file_size_bytes >= 1024 * 1024:
        size_str = f"{info.file_size_bytes / (1024**2):.2f} MB"
    elif info.file_size_bytes >= 1024:
        size_str = f"{info.file_size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{info.file_size_bytes} bytes"

    # Format sample rate
    if info.sample_rate >= 1e6:
        rate_str = f"{info.sample_rate / 1e6:.3f} MHz"
    elif info.sample_rate >= 1e3:
        rate_str = f"{info.sample_rate / 1e3:.3f} kHz"
    elif info.sample_rate > 0:
        rate_str = f"{info.sample_rate:.1f} Hz"
    else:
        rate_str = "unknown"

    # Format center frequency
    if info.center_frequency >= 1e9:
        freq_str = f"{info.center_frequency / 1e9:.6f} GHz"
    elif info.center_frequency >= 1e6:
        freq_str = f"{info.center_frequency / 1e6:.6f} MHz"
    elif info.center_frequency >= 1e3:
        freq_str = f"{info.center_frequency / 1e3:.3f} kHz"
    elif info.center_frequency > 0:
        freq_str = f"{info.center_frequency:.1f} Hz"
    else:
        freq_str = "not specified"

    return {
        "file_format": info.file_format.value,
        "sample_format": info.sample_format.value,
        "sample_rate": rate_str,
        "sample_rate_hz": info.sample_rate,
        "center_frequency": freq_str,
        "center_frequency_hz": info.center_frequency,
        "duration": duration_str,
        "duration_seconds": info.duration_seconds,
        "num_samples": info.num_samples,
        "file_size": size_str,
        "file_size_bytes": info.file_size_bytes,
        "bits_per_sample": info.bits_per_sample,
        "is_valid": info.is_valid,
        "error": info.error_message if not info.is_valid else None,
        "metadata": info.metadata,
    }

"""
I/Q Recording and Playback for SDR signals.

Provides functionality to record and playback raw baseband signals:
- Multiple file formats: raw I/Q, WAV, SigMF
- Various sample formats: 8-bit, 16-bit, 32-bit float
- Streaming recording with buffering
- Metadata support (frequency, sample rate, timestamps)
"""

import numpy as np
import json
import wave
import struct
import os
from typing import Optional, Tuple, Dict, Any, BinaryIO, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


class SampleFormat(Enum):
    """Sample format for I/Q data."""
    UINT8 = "cu8"        # Unsigned 8-bit (RTL-SDR native)
    INT8 = "cs8"         # Signed 8-bit
    INT16 = "cs16"       # Signed 16-bit (common)
    FLOAT32 = "cf32"     # 32-bit float complex
    FLOAT64 = "cf64"     # 64-bit float complex


class FileFormat(Enum):
    """Recording file format."""
    RAW = "raw"          # Raw I/Q binary
    WAV = "wav"          # WAV file (audio compatible)
    SIGMF = "sigmf"      # Signal Metadata Format


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
        buffer_size: int = 65536
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
            sample_format=sample_format
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
            self._file = open(self._filepath, 'wb')

        self._recording = True

    def _start_wav(self) -> None:
        """Start WAV file recording."""
        self._wav_file = wave.open(str(self._filepath), 'wb')

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
        else:
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
            "annotations": []
        }

        # Add optional fields
        if self._metadata.hardware:
            sigmf_meta["global"]["core:hw"] = self._metadata.hardware

        # Write metadata file
        meta_path = self._filepath.with_suffix('.sigmf-meta')
        with open(meta_path, 'w') as f:
            json.dump(sigmf_meta, f, indent=2)

        # Rename data file for SigMF
        data_path = self._filepath.with_suffix('.sigmf-data')
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
        sample_rate: Optional[float] = None
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
        if suffix in ('.sigmf-data', '.sigmf'):
            self._open_sigmf()
        elif suffix == '.wav':
            self._open_wav()
        else:
            self._open_raw()

    def _open_sigmf(self) -> None:
        """Open SigMF recording."""
        # Find metadata file
        meta_path = self._filepath.with_suffix('.sigmf-meta')
        data_path = self._filepath.with_suffix('.sigmf-data')

        if not data_path.exists():
            data_path = self._filepath

        # Read metadata
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                sigmf_meta = json.load(f)

            global_meta = sigmf_meta.get('global', {})
            captures = sigmf_meta.get('captures', [{}])

            # Parse datatype
            datatype = global_meta.get('core:datatype', 'cf32_le')
            format_map = {
                'cu8': SampleFormat.UINT8,
                'ci8': SampleFormat.INT8,
                'ci16_le': SampleFormat.INT16,
                'cf32_le': SampleFormat.FLOAT32,
                'cf64_le': SampleFormat.FLOAT64,
            }
            self._sample_format = format_map.get(datatype, SampleFormat.FLOAT32)

            self._sample_rate = global_meta.get('core:sample_rate', 0.0)
            if captures:
                self._center_frequency = captures[0].get('core:frequency', 0.0)

            self._metadata = RecordingMetadata(
                sample_rate=self._sample_rate,
                center_frequency=self._center_frequency,
                sample_format=self._sample_format,
                description=global_meta.get('core:description', ''),
                hardware=global_meta.get('core:hw', ''),
            )

        # Open data file
        self._file = open(data_path, 'rb')

        # Calculate total samples
        file_size = data_path.stat().st_size
        bytes_per_sample = self._get_bytes_per_sample() * 2  # I and Q
        self._total_samples = file_size // bytes_per_sample

    def _open_wav(self) -> None:
        """Open WAV recording."""
        self._wav_file = wave.open(str(self._filepath), 'rb')

        # Get parameters
        n_channels = self._wav_file.getnchannels()
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
            '.cu8': SampleFormat.UINT8,
            '.cs8': SampleFormat.INT8,
            '.cs16': SampleFormat.INT16,
            '.cf32': SampleFormat.FLOAT32,
            '.cf64': SampleFormat.FLOAT64,
            '.raw': SampleFormat.FLOAT32,
            '.iq': SampleFormat.FLOAT32,
        }

        if self._sample_format is None:
            self._sample_format = format_map.get(suffix, SampleFormat.FLOAT32)

        self._file = open(self._filepath, 'rb')

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
            n_complex = len(samples) // 2
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
        prefix: str = "recording"
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
            file_format=self._file_format
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
    max_samples: Optional[int] = None
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
    file_format: FileFormat = FileFormat.RAW
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
        file_format=file_format
    )

    recorder.start(filepath)
    recorder.write(samples)
    return recorder.stop()

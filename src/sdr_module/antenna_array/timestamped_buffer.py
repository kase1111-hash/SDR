"""
Timestamped sample buffer for antenna array synchronization.

Extends the base SampleBuffer with timing metadata to enable
phase-coherent processing across multiple SDR devices.
"""

import time
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Optional, Tuple

import numpy as np

from ..core.sample_buffer import BufferOverflowPolicy, BufferStats


@dataclass
class TimestampedChunk:
    """A chunk of samples with associated timing metadata."""

    samples: np.ndarray
    timestamp: float  # System time when samples were received
    sample_index: int  # Running index of first sample in chunk
    sample_rate: float  # Sample rate at time of capture
    device_id: str  # Identifier of source device

    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate

    @property
    def end_timestamp(self) -> float:
        """Estimated timestamp at end of chunk."""
        return self.timestamp + self.duration

    @property
    def end_sample_index(self) -> int:
        """Sample index at end of chunk."""
        return self.sample_index + len(self.samples)


@dataclass
class TimestampedBufferStats(BufferStats):
    """Extended statistics for timestamped buffer."""

    total_chunks_in: int = 0
    total_chunks_out: int = 0
    chunks_dropped: int = 0
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    timestamp_discontinuities: int = 0

    @property
    def time_span(self) -> float:
        """Total time span of buffered data."""
        if self.first_timestamp == 0.0:
            return 0.0
        return self.last_timestamp - self.first_timestamp


class TimestampedSampleBuffer:
    """
    Thread-safe circular buffer for timestamped I/Q samples.

    Designed for antenna array applications where timing information
    is critical for phase alignment and synchronization.

    Unlike the base SampleBuffer which stores raw samples, this buffer
    stores TimestampedChunk objects that preserve timing metadata.

    Thread Safety:
        - All public methods are thread-safe (protected by internal Lock)
        - write() can be called from producer threads (device callbacks)
        - read()/peek() can be called from consumer thread (DSP pipeline)
        - stats property returns a thread-safe copy
        - Uses Condition variables for efficient blocking
    """

    def __init__(
        self,
        capacity_chunks: int = 256,
        capacity_samples: int = 4 * 1024 * 1024,  # 4M samples
        overflow_policy: BufferOverflowPolicy = BufferOverflowPolicy.DROP_OLDEST,
        device_id: str = "",
    ) -> None:
        """
        Initialize timestamped sample buffer.

        Args:
            capacity_chunks: Maximum number of chunks to store
            capacity_samples: Maximum number of samples across all chunks
            overflow_policy: How to handle buffer overflow
            device_id: Identifier for this buffer's source device
        """
        self._capacity_chunks = capacity_chunks
        self._capacity_samples = capacity_samples
        self._overflow_policy = overflow_policy
        self._device_id = device_id

        # Chunk storage (circular buffer of chunks)
        self._chunks: list[Optional[TimestampedChunk]] = [None] * capacity_chunks
        self._write_idx = 0
        self._read_idx = 0
        self._chunk_count = 0
        self._sample_count = 0

        # Running sample index counter
        self._next_sample_index = 0

        # Thread synchronization
        self._lock = Lock()
        self._not_empty = Condition(self._lock)
        self._not_full = Condition(self._lock)

        # Statistics
        self._stats = TimestampedBufferStats(capacity=capacity_samples)

        # Phase tracking
        self._last_phase: Optional[float] = None
        self._phase_offset: float = 0.0  # Calibration offset

    @property
    def device_id(self) -> str:
        """Get device identifier."""
        return self._device_id

    @property
    def capacity_chunks(self) -> int:
        """Maximum number of chunks."""
        return self._capacity_chunks

    @property
    def capacity_samples(self) -> int:
        """Maximum number of samples."""
        return self._capacity_samples

    @property
    def available_chunks(self) -> int:
        """Number of chunks available to read."""
        with self._lock:
            return self._chunk_count

    @property
    def available_samples(self) -> int:
        """Total number of samples available."""
        with self._lock:
            return self._sample_count

    @property
    def stats(self) -> TimestampedBufferStats:
        """Get buffer statistics (thread-safe copy)."""
        with self._lock:
            return TimestampedBufferStats(
                total_samples_in=self._stats.total_samples_in,
                total_samples_out=self._stats.total_samples_out,
                samples_dropped=self._stats.samples_dropped,
                current_fill=self._sample_count,
                capacity=self._capacity_samples,
                overflow_count=self._stats.overflow_count,
                total_chunks_in=self._stats.total_chunks_in,
                total_chunks_out=self._stats.total_chunks_out,
                chunks_dropped=self._stats.chunks_dropped,
                first_timestamp=self._stats.first_timestamp,
                last_timestamp=self._stats.last_timestamp,
                timestamp_discontinuities=self._stats.timestamp_discontinuities,
            )

    @property
    def phase_offset(self) -> float:
        """Get phase calibration offset in radians."""
        with self._lock:
            return self._phase_offset

    @phase_offset.setter
    def phase_offset(self, offset: float) -> None:
        """Set phase calibration offset in radians."""
        with self._lock:
            self._phase_offset = offset

    def write(
        self,
        samples: np.ndarray,
        timestamp: Optional[float] = None,
        sample_rate: float = 0.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Write samples to buffer with timestamp.

        Args:
            samples: Complex samples to write
            timestamp: System timestamp (uses current time if None)
            sample_rate: Sample rate of the data
            timeout: Timeout in seconds for BLOCK policy

        Returns:
            True if samples were written, False on timeout/drop
        """
        if len(samples) == 0:
            return True

        # Ensure complex64
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64)

        # Use current time if not provided
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Check for timestamp discontinuity
            if self._stats.last_timestamp > 0:
                expected_gap = len(samples) / sample_rate if sample_rate > 0 else 0
                actual_gap = timestamp - self._stats.last_timestamp
                # Allow 10% tolerance + 10ms for system jitter
                if sample_rate > 0 and abs(actual_gap - expected_gap) > (
                    expected_gap * 0.1 + 0.01
                ):
                    self._stats.timestamp_discontinuities += 1

            # Handle overflow for chunks
            if self._chunk_count >= self._capacity_chunks:
                if self._overflow_policy == BufferOverflowPolicy.BLOCK:
                    while self._chunk_count >= self._capacity_chunks:
                        if not self._not_full.wait(timeout):
                            return False
                elif self._overflow_policy == BufferOverflowPolicy.DROP_NEWEST:
                    self._stats.chunks_dropped += 1
                    self._stats.samples_dropped += len(samples)
                    self._stats.overflow_count += 1
                    return False
                else:  # DROP_OLDEST
                    # Remove oldest chunk
                    oldest = self._chunks[self._read_idx]
                    if oldest is not None:
                        self._sample_count -= len(oldest.samples)
                        self._stats.samples_dropped += len(oldest.samples)
                    self._chunks[self._read_idx] = None
                    self._read_idx = (self._read_idx + 1) % self._capacity_chunks
                    self._chunk_count -= 1
                    self._stats.chunks_dropped += 1
                    self._stats.overflow_count += 1

            # Handle overflow for samples
            while (
                self._sample_count + len(samples) > self._capacity_samples
                and self._chunk_count > 0
            ):
                if self._overflow_policy == BufferOverflowPolicy.BLOCK:
                    if not self._not_full.wait(timeout):
                        return False
                elif self._overflow_policy == BufferOverflowPolicy.DROP_NEWEST:
                    self._stats.samples_dropped += len(samples)
                    self._stats.overflow_count += 1
                    return False
                else:  # DROP_OLDEST
                    oldest = self._chunks[self._read_idx]
                    if oldest is not None:
                        self._sample_count -= len(oldest.samples)
                        self._stats.samples_dropped += len(oldest.samples)
                    self._chunks[self._read_idx] = None
                    self._read_idx = (self._read_idx + 1) % self._capacity_chunks
                    self._chunk_count -= 1
                    self._stats.chunks_dropped += 1
                    self._stats.overflow_count += 1

            # Create timestamped chunk
            chunk = TimestampedChunk(
                samples=samples.copy(),
                timestamp=timestamp,
                sample_index=self._next_sample_index,
                sample_rate=sample_rate,
                device_id=self._device_id,
            )

            # Write chunk
            self._chunks[self._write_idx] = chunk
            self._write_idx = (self._write_idx + 1) % self._capacity_chunks
            self._chunk_count += 1
            self._sample_count += len(samples)
            self._next_sample_index += len(samples)

            # Update statistics
            self._stats.total_samples_in += len(samples)
            self._stats.total_chunks_in += 1
            if self._stats.first_timestamp == 0.0:
                self._stats.first_timestamp = timestamp
            self._stats.last_timestamp = timestamp

            # Signal readers
            self._not_empty.notify_all()

            return True

    def read(self, timeout: Optional[float] = None) -> Optional[TimestampedChunk]:
        """
        Read the oldest chunk from buffer.

        Args:
            timeout: Timeout in seconds (None = block indefinitely)

        Returns:
            TimestampedChunk or None on timeout
        """
        with self._lock:
            while self._chunk_count == 0:
                if timeout == 0:
                    return None
                if not self._not_empty.wait(timeout):
                    return None

            chunk = self._chunks[self._read_idx]
            self._chunks[self._read_idx] = None
            self._read_idx = (self._read_idx + 1) % self._capacity_chunks
            self._chunk_count -= 1

            if chunk is not None:
                self._sample_count -= len(chunk.samples)
                self._stats.total_samples_out += len(chunk.samples)
                self._stats.total_chunks_out += 1

            # Signal writers
            self._not_full.notify_all()

            return chunk

    def read_samples(
        self, n_samples: int, timeout: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Read a specific number of samples, potentially spanning multiple chunks.

        Args:
            n_samples: Number of samples to read
            timeout: Timeout in seconds

        Returns:
            Tuple of (samples, first_timestamp, first_sample_index) or None
        """
        with self._lock:
            # Wait for enough samples
            while self._sample_count < n_samples:
                if timeout == 0:
                    return None
                if not self._not_empty.wait(timeout):
                    return None

            # Collect samples from chunks
            collected: list[np.ndarray] = []
            first_timestamp: Optional[float] = None
            first_sample_index: Optional[int] = None
            remaining = n_samples

            while remaining > 0 and self._chunk_count > 0:
                chunk = self._chunks[self._read_idx]
                if chunk is None:
                    break

                if first_timestamp is None:
                    first_timestamp = chunk.timestamp
                    first_sample_index = chunk.sample_index

                if len(chunk.samples) <= remaining:
                    # Take entire chunk
                    collected.append(chunk.samples)
                    remaining -= len(chunk.samples)
                    self._sample_count -= len(chunk.samples)
                    self._stats.total_samples_out += len(chunk.samples)
                    self._stats.total_chunks_out += 1
                    self._chunks[self._read_idx] = None
                    self._read_idx = (self._read_idx + 1) % self._capacity_chunks
                    self._chunk_count -= 1
                else:
                    # Take partial chunk
                    collected.append(chunk.samples[:remaining])
                    # Update chunk in place
                    new_chunk = TimestampedChunk(
                        samples=chunk.samples[remaining:],
                        timestamp=chunk.timestamp
                        + remaining / chunk.sample_rate
                        if chunk.sample_rate > 0
                        else chunk.timestamp,
                        sample_index=chunk.sample_index + remaining,
                        sample_rate=chunk.sample_rate,
                        device_id=chunk.device_id,
                    )
                    self._chunks[self._read_idx] = new_chunk
                    self._sample_count -= remaining
                    self._stats.total_samples_out += remaining
                    remaining = 0

            self._not_full.notify_all()

            if not collected:
                return None

            samples = np.concatenate(collected) if len(collected) > 1 else collected[0]
            return (samples, first_timestamp or 0.0, first_sample_index or 0)

    def peek(self) -> Optional[TimestampedChunk]:
        """
        Peek at the oldest chunk without removing it.

        Returns:
            TimestampedChunk or None if empty
        """
        with self._lock:
            if self._chunk_count == 0:
                return None
            chunk = self._chunks[self._read_idx]
            if chunk is None:
                return None
            # Return a copy to prevent external modification
            return TimestampedChunk(
                samples=chunk.samples.copy(),
                timestamp=chunk.timestamp,
                sample_index=chunk.sample_index,
                sample_rate=chunk.sample_rate,
                device_id=chunk.device_id,
            )

    def get_timestamp_range(self) -> Tuple[float, float]:
        """
        Get the timestamp range of buffered data.

        Returns:
            Tuple of (first_timestamp, last_timestamp)
        """
        with self._lock:
            return (self._stats.first_timestamp, self._stats.last_timestamp)

    def get_sample_index_range(self) -> Tuple[int, int]:
        """
        Get the sample index range of buffered data.

        Returns:
            Tuple of (first_sample_index, last_sample_index)
        """
        with self._lock:
            if self._chunk_count == 0:
                return (0, 0)

            first_chunk = self._chunks[self._read_idx]
            # Find last valid chunk
            last_idx = (self._write_idx - 1) % self._capacity_chunks
            last_chunk = self._chunks[last_idx]

            first_index = first_chunk.sample_index if first_chunk else 0
            last_index = last_chunk.end_sample_index if last_chunk else 0

            return (first_index, last_index)

    def clear(self) -> None:
        """Clear all chunks from buffer."""
        with self._lock:
            for i in range(self._capacity_chunks):
                self._chunks[i] = None
            self._write_idx = 0
            self._read_idx = 0
            self._chunk_count = 0
            self._sample_count = 0
            self._not_full.notify_all()
            self._not_empty.notify_all()

    def reset_stats(self) -> None:
        """Reset buffer statistics."""
        with self._lock:
            self._stats = TimestampedBufferStats(
                capacity=self._capacity_samples, current_fill=self._sample_count
            )

    def reset_sample_index(self, start_index: int = 0) -> None:
        """
        Reset the sample index counter.

        Useful for synchronizing indices across multiple buffers.

        Args:
            start_index: New starting index
        """
        with self._lock:
            self._next_sample_index = start_index

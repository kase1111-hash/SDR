"""
Sample buffer for efficient I/Q data handling.

Provides thread-safe circular buffers for streaming samples
between device drivers and DSP processing chains.
"""

from dataclasses import dataclass
from enum import Enum
from threading import Condition, Lock
from typing import Optional

import numpy as np


class BufferOverflowPolicy(Enum):
    """Policy for handling buffer overflow."""

    DROP_OLDEST = "drop_oldest"  # Drop oldest samples (default)
    DROP_NEWEST = "drop_newest"  # Drop incoming samples
    BLOCK = "block"  # Block until space available


@dataclass
class BufferStats:
    """Buffer statistics."""

    total_samples_in: int = 0
    total_samples_out: int = 0
    samples_dropped: int = 0
    current_fill: int = 0
    capacity: int = 0
    overflow_count: int = 0

    @property
    def fill_ratio(self) -> float:
        """Current fill ratio (0.0 to 1.0)."""
        if self.capacity == 0:
            return 0.0
        return self.current_fill / self.capacity


class SampleBuffer:
    """
    Thread-safe circular buffer for complex I/Q samples.

    Designed for producer-consumer pattern between SDR device
    and DSP processing pipeline.
    """

    def __init__(
        self,
        capacity: int = 1024 * 1024,  # 1M samples default
        overflow_policy: BufferOverflowPolicy = BufferOverflowPolicy.DROP_OLDEST,
    ):
        """
        Initialize sample buffer.

        Args:
            capacity: Maximum number of complex samples
            overflow_policy: How to handle buffer overflow
        """
        self._capacity = capacity
        self._overflow_policy = overflow_policy

        # Circular buffer storage
        self._buffer = np.zeros(capacity, dtype=np.complex64)
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0

        # Thread synchronization
        self._lock = Lock()
        self._not_empty = Condition(self._lock)
        self._not_full = Condition(self._lock)

        # Statistics
        self._stats = BufferStats(capacity=capacity)

    @property
    def capacity(self) -> int:
        """Buffer capacity in samples."""
        return self._capacity

    @property
    def available(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self._count

    @property
    def free_space(self) -> int:
        """Number of samples that can be written."""
        with self._lock:
            return self._capacity - self._count

    @property
    def stats(self) -> BufferStats:
        """Get buffer statistics."""
        with self._lock:
            self._stats.current_fill = self._count
            return BufferStats(
                total_samples_in=self._stats.total_samples_in,
                total_samples_out=self._stats.total_samples_out,
                samples_dropped=self._stats.samples_dropped,
                current_fill=self._count,
                capacity=self._capacity,
                overflow_count=self._stats.overflow_count,
            )

    def write(self, samples: np.ndarray, timeout: Optional[float] = None) -> int:
        """
        Write samples to buffer.

        Args:
            samples: Complex samples to write
            timeout: Timeout in seconds (None = non-blocking for DROP policies)

        Returns:
            Number of samples actually written
        """
        if len(samples) == 0:
            return 0

        # Ensure complex64
        if samples.dtype != np.complex64:
            samples = samples.astype(np.complex64)

        with self._lock:
            n_samples = len(samples)

            # If samples exceed capacity, truncate to capacity
            if n_samples > self._capacity:
                self._stats.samples_dropped += n_samples - self._capacity
                self._stats.overflow_count += 1
                samples = samples[-self._capacity :]  # Keep newest
                n_samples = self._capacity

            # Handle overflow
            if n_samples > self._capacity - self._count:
                if self._overflow_policy == BufferOverflowPolicy.BLOCK:
                    # Wait for space
                    while self._capacity - self._count < n_samples:
                        if not self._not_full.wait(timeout):
                            # Timeout, write what we can
                            n_samples = self._capacity - self._count
                            break
                elif self._overflow_policy == BufferOverflowPolicy.DROP_NEWEST:
                    # Only write what fits
                    n_samples = self._capacity - self._count
                    self._stats.samples_dropped += len(samples) - n_samples
                    self._stats.overflow_count += 1
                else:  # DROP_OLDEST
                    # Make room by advancing read pointer
                    overflow = n_samples - (self._capacity - self._count)
                    self._read_idx = (self._read_idx + overflow) % self._capacity
                    self._count -= overflow
                    self._stats.samples_dropped += overflow
                    self._stats.overflow_count += 1

            if n_samples == 0:
                return 0

            # Write samples (may wrap around)
            samples = samples[:n_samples]
            end_idx = (self._write_idx + n_samples) % self._capacity

            if end_idx > self._write_idx:
                # No wrap
                self._buffer[self._write_idx : end_idx] = samples
            else:
                # Wrap around
                first_chunk = self._capacity - self._write_idx
                self._buffer[self._write_idx :] = samples[:first_chunk]
                if end_idx > 0:
                    self._buffer[:end_idx] = samples[first_chunk:]

            self._write_idx = end_idx
            self._count += n_samples
            self._stats.total_samples_in += n_samples

            # Signal readers
            self._not_empty.notify_all()

            return n_samples

    def read(
        self, n_samples: int, timeout: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Read samples from buffer.

        Args:
            n_samples: Number of samples to read
            timeout: Timeout in seconds (None = block indefinitely)

        Returns:
            Complex numpy array, or None on timeout
        """
        with self._lock:
            # Wait for samples
            while self._count < n_samples:
                if timeout == 0:
                    # Non-blocking, return what's available
                    n_samples = self._count
                    if n_samples == 0:
                        return None
                    break
                if not self._not_empty.wait(timeout):
                    # Timeout
                    return None

            if n_samples == 0:
                return np.array([], dtype=np.complex64)

            # Read samples (may wrap around)
            end_idx = (self._read_idx + n_samples) % self._capacity

            if end_idx > self._read_idx:
                # No wrap
                samples = self._buffer[self._read_idx : end_idx].copy()
            else:
                # Wrap around
                samples = np.concatenate(
                    [self._buffer[self._read_idx :], self._buffer[:end_idx]]
                )

            self._read_idx = end_idx
            self._count -= n_samples
            self._stats.total_samples_out += n_samples

            # Signal writers
            self._not_full.notify_all()

            return samples

    def peek(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Peek at samples without removing them.

        Args:
            n_samples: Number of samples to peek

        Returns:
            Complex numpy array, or None if not enough samples
        """
        with self._lock:
            if self._count < n_samples:
                return None

            end_idx = (self._read_idx + n_samples) % self._capacity

            if end_idx > self._read_idx:
                return self._buffer[self._read_idx : end_idx].copy()
            else:
                return np.concatenate(
                    [self._buffer[self._read_idx :], self._buffer[:end_idx]]
                )

    def clear(self) -> None:
        """Clear all samples from buffer."""
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            self._not_full.notify_all()

    def reset_stats(self) -> None:
        """Reset buffer statistics."""
        with self._lock:
            self._stats = BufferStats(capacity=self._capacity, current_fill=self._count)

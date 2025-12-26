"""Tests for sample buffer implementation."""

import pytest
import numpy as np
import threading
import time
from sdr_module.core.sample_buffer import (
    SampleBuffer,
    BufferOverflowPolicy,
    BufferStats,
)


class TestSampleBuffer:
    """Test basic buffer operations."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = SampleBuffer(capacity=1024)
        assert buffer.capacity == 1024
        assert buffer.available == 0
        assert buffer.free_space == 1024

    def test_write_read_simple(self):
        """Test simple write and read."""
        buffer = SampleBuffer(capacity=1024)
        samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)

        written = buffer.write(samples)
        assert written == 3
        assert buffer.available == 3

        read_samples = buffer.read(3, timeout=0)
        assert read_samples is not None
        assert len(read_samples) == 3
        np.testing.assert_array_equal(read_samples, samples)

    def test_write_read_large(self):
        """Test write and read with larger data."""
        buffer = SampleBuffer(capacity=10000)
        samples = np.random.randn(5000) + 1j * np.random.randn(5000)
        samples = samples.astype(np.complex64)

        buffer.write(samples)
        assert buffer.available == 5000

        read_samples = buffer.read(5000, timeout=0.1)
        np.testing.assert_array_almost_equal(read_samples, samples)

    def test_multiple_writes_reads(self):
        """Test multiple write and read operations."""
        buffer = SampleBuffer(capacity=1000)

        for i in range(5):
            samples = np.ones(100, dtype=np.complex64) * (i + 1)
            buffer.write(samples)

        assert buffer.available == 500

        for i in range(5):
            samples = buffer.read(100, timeout=0)
            expected = np.ones(100, dtype=np.complex64) * (i + 1)
            np.testing.assert_array_equal(samples, expected)

    def test_circular_wrap(self):
        """Test circular buffer wraparound."""
        buffer = SampleBuffer(capacity=100)

        # Fill buffer
        samples1 = np.ones(80, dtype=np.complex64)
        buffer.write(samples1)

        # Read some
        buffer.read(60, timeout=0)

        # Write more (will wrap around)
        samples2 = np.ones(70, dtype=np.complex64) * 2
        buffer.write(samples2)

        # Read remaining from first write
        result1 = buffer.read(20, timeout=0)
        np.testing.assert_array_equal(result1, np.ones(20, dtype=np.complex64))

        # Read from second write
        result2 = buffer.read(70, timeout=0)
        np.testing.assert_array_equal(result2, np.ones(70, dtype=np.complex64) * 2)


class TestBufferOverflow:
    """Test buffer overflow policies."""

    def test_drop_oldest_policy(self):
        """Test DROP_OLDEST overflow policy."""
        buffer = SampleBuffer(
            capacity=100, overflow_policy=BufferOverflowPolicy.DROP_OLDEST
        )

        # Fill buffer
        samples1 = np.ones(100, dtype=np.complex64)
        buffer.write(samples1)

        # Overflow with new data
        samples2 = np.ones(50, dtype=np.complex64) * 2
        written = buffer.write(samples2)

        assert written == 50
        assert buffer.available == 100

        # Should have dropped oldest 50
        result = buffer.read(100, timeout=0)
        assert np.all(result[:50] == 1.0)  # Last 50 from first write
        assert np.all(result[50:] == 2.0)  # All of second write

    def test_drop_newest_policy(self):
        """Test DROP_NEWEST overflow policy."""
        buffer = SampleBuffer(
            capacity=100, overflow_policy=BufferOverflowPolicy.DROP_NEWEST
        )

        # Fill buffer
        samples1 = np.ones(100, dtype=np.complex64)
        buffer.write(samples1)

        # Try to overflow
        samples2 = np.ones(50, dtype=np.complex64) * 2
        written = buffer.write(samples2)

        assert written == 0  # Nothing written
        assert buffer.available == 100

        # Should still have original data
        result = buffer.read(100, timeout=0)
        np.testing.assert_array_equal(result, samples1)

    def test_block_policy(self):
        """Test BLOCK overflow policy with timeout."""
        buffer = SampleBuffer(
            capacity=100, overflow_policy=BufferOverflowPolicy.BLOCK
        )

        # Fill buffer
        samples1 = np.ones(100, dtype=np.complex64)
        buffer.write(samples1)

        # Try to write more with short timeout
        samples2 = np.ones(50, dtype=np.complex64) * 2
        written = buffer.write(samples2, timeout=0.1)

        # Should have written nothing or partial
        assert written <= 50


class TestBufferStats:
    """Test buffer statistics."""

    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        buffer = SampleBuffer(capacity=1000)

        samples = np.ones(100, dtype=np.complex64)
        buffer.write(samples)
        buffer.write(samples)

        stats = buffer.stats
        assert stats.total_samples_in == 200
        assert stats.current_fill == 200
        assert stats.capacity == 1000
        assert stats.fill_ratio == pytest.approx(0.2)

        buffer.read(150, timeout=0)
        stats = buffer.stats
        assert stats.total_samples_out == 150
        assert stats.current_fill == 50

    def test_overflow_stats(self):
        """Test overflow statistics."""
        buffer = SampleBuffer(
            capacity=100, overflow_policy=BufferOverflowPolicy.DROP_OLDEST
        )

        # Fill buffer
        buffer.write(np.ones(100, dtype=np.complex64))

        # Overflow
        buffer.write(np.ones(50, dtype=np.complex64))

        stats = buffer.stats
        assert stats.samples_dropped == 50
        assert stats.overflow_count == 1


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_write_read(self):
        """Test concurrent writes and reads."""
        buffer = SampleBuffer(capacity=10000)
        write_count = 0
        read_count = 0
        errors = []

        def writer():
            nonlocal write_count
            try:
                for _ in range(100):
                    samples = np.random.randn(50) + 1j * np.random.randn(50)
                    samples = samples.astype(np.complex64)
                    buffer.write(samples)
                    write_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            nonlocal read_count
            try:
                for _ in range(100):
                    samples = buffer.read(50, timeout=1.0)
                    if samples is not None:
                        read_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start threads
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join(timeout=10)
        reader_thread.join(timeout=10)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert write_count == 100
        assert read_count > 0  # Should have read some


class TestBufferMethods:
    """Test additional buffer methods."""

    def test_peek(self):
        """Test peek operation."""
        buffer = SampleBuffer(capacity=1000)
        samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
        buffer.write(samples)

        # Peek shouldn't remove data
        peeked = buffer.peek(3)
        assert peeked is not None
        np.testing.assert_array_equal(peeked, samples)
        assert buffer.available == 3

        # Can still read
        read_samples = buffer.read(3, timeout=0)
        np.testing.assert_array_equal(read_samples, samples)
        assert buffer.available == 0

    def test_peek_insufficient_data(self):
        """Test peek with insufficient data."""
        buffer = SampleBuffer(capacity=1000)
        buffer.write(np.ones(5, dtype=np.complex64))

        result = buffer.peek(10)
        assert result is None

    def test_clear(self):
        """Test buffer clear."""
        buffer = SampleBuffer(capacity=1000)
        buffer.write(np.ones(100, dtype=np.complex64))

        assert buffer.available == 100
        buffer.clear()
        assert buffer.available == 0
        assert buffer.free_space == 1000

    def test_reset_stats(self):
        """Test statistics reset."""
        buffer = SampleBuffer(capacity=1000)
        buffer.write(np.ones(100, dtype=np.complex64))
        buffer.read(50, timeout=0)

        buffer.reset_stats()
        stats = buffer.stats
        assert stats.total_samples_in == 0
        assert stats.total_samples_out == 0
        assert stats.current_fill == 50  # Data still there


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_write(self):
        """Test writing empty array."""
        buffer = SampleBuffer(capacity=1000)
        written = buffer.write(np.array([], dtype=np.complex64))
        assert written == 0

    def test_read_timeout(self):
        """Test read timeout when buffer is empty."""
        buffer = SampleBuffer(capacity=1000)
        result = buffer.read(10, timeout=0.1)
        assert result is None

    def test_read_nonblocking_empty(self):
        """Test non-blocking read from empty buffer."""
        buffer = SampleBuffer(capacity=1000)
        result = buffer.read(10, timeout=0)
        assert result is None

    def test_read_partial_nonblocking(self):
        """Test non-blocking read returns partial data."""
        buffer = SampleBuffer(capacity=1000)
        buffer.write(np.ones(5, dtype=np.complex64))

        result = buffer.read(10, timeout=0)
        assert result is not None
        assert len(result) == 5

    def test_type_conversion(self):
        """Test automatic type conversion to complex64."""
        buffer = SampleBuffer(capacity=1000)

        # Write complex128
        samples = np.ones(10, dtype=np.complex128)
        buffer.write(samples)

        result = buffer.read(10, timeout=0)
        assert result.dtype == np.complex64

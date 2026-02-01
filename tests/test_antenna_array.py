"""Tests for antenna array module implementation."""

import math
import threading
import time

import numpy as np
import pytest

from sdr_module.antenna_array import (
    SPEED_OF_LIGHT,
    AntennaArrayController,
    ArrayConfig,
    ArrayElement,
    ArrayGeometry,
    ArrayOperationMode,
    ElementCalibration,
    ElementPosition,
    SynchronizationConfig,
    SyncState,
    TimestampedChunk,
    TimestampedSampleBuffer,
    TimestampedBufferStats,
    create_circular_4_element,
    create_linear_2_element,
    create_linear_4_element,
    create_rectangular_2x2,
    get_array_preset,
    list_array_presets,
)
from sdr_module.core.sample_buffer import BufferOverflowPolicy


class TestElementPosition:
    """Test ElementPosition class."""

    def test_initialization(self):
        """Test position initialization."""
        pos = ElementPosition(x=1.0, y=2.0, z=3.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0

    def test_default_values(self):
        """Test default position at origin."""
        pos = ElementPosition()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.z == 0.0

    def test_to_array(self):
        """Test conversion to numpy array."""
        pos = ElementPosition(x=1.0, y=2.0, z=3.0)
        arr = pos.to_array()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
        assert arr.dtype == np.float64

    def test_from_array(self):
        """Test creation from numpy array."""
        arr = np.array([4.0, 5.0, 6.0])
        pos = ElementPosition.from_array(arr)
        assert pos.x == 4.0
        assert pos.y == 5.0
        assert pos.z == 6.0

    def test_distance_to(self):
        """Test distance calculation."""
        pos1 = ElementPosition(x=0.0, y=0.0, z=0.0)
        pos2 = ElementPosition(x=3.0, y=4.0, z=0.0)
        assert pos1.distance_to(pos2) == pytest.approx(5.0)

    def test_distance_to_self(self):
        """Test distance to self is zero."""
        pos = ElementPosition(x=1.0, y=2.0, z=3.0)
        assert pos.distance_to(pos) == pytest.approx(0.0)


class TestElementCalibration:
    """Test ElementCalibration class."""

    def test_initialization(self):
        """Test calibration initialization."""
        cal = ElementCalibration(
            element_index=0,
            phase_offset=0.5,
            amplitude_scale=1.1,
            delay_samples=2.5,
        )
        assert cal.element_index == 0
        assert cal.phase_offset == 0.5
        assert cal.amplitude_scale == 1.1
        assert cal.delay_samples == 2.5

    def test_get_correction_phasor(self):
        """Test correction phasor calculation."""
        cal = ElementCalibration(phase_offset=np.pi / 2, amplitude_scale=2.0)
        phasor = cal.get_correction_phasor()

        # Should be amplitude * exp(-j * phase)
        expected = 2.0 * np.exp(-1j * np.pi / 2)
        assert np.isclose(phasor, expected)

    def test_default_correction_is_unity(self):
        """Test default calibration gives unity correction."""
        cal = ElementCalibration()
        phasor = cal.get_correction_phasor()
        assert np.isclose(phasor, 1.0 + 0j)


class TestArrayGeometry:
    """Test ArrayGeometry class."""

    def test_linear_geometry(self):
        """Test linear array generation."""
        geom = ArrayGeometry(
            geometry_type="linear",
            num_elements=4,
            element_spacing=0.5,
            reference_frequency=300e6,
        )
        assert len(geom.elements) == 4

        # Check elements are along X axis
        for pos in geom.elements:
            assert pos.y == 0.0
            assert pos.z == 0.0

        # Check spacing
        wavelength = SPEED_OF_LIGHT / 300e6
        expected_spacing = 0.5 * wavelength
        actual_spacing = geom.elements[1].x - geom.elements[0].x
        assert actual_spacing == pytest.approx(expected_spacing)

    def test_circular_geometry(self):
        """Test circular array generation."""
        geom = ArrayGeometry(
            geometry_type="circular",
            num_elements=4,
            element_spacing=0.5,
            reference_frequency=433e6,
        )
        assert len(geom.elements) == 4

        # All elements should be at same height
        for pos in geom.elements:
            assert pos.z == 0.0

        # All elements should be equidistant from center
        distances = [math.sqrt(pos.x**2 + pos.y**2) for pos in geom.elements]
        for d in distances:
            assert d == pytest.approx(distances[0])

    def test_rectangular_geometry(self):
        """Test rectangular array generation."""
        geom = ArrayGeometry(
            geometry_type="rectangular",
            num_elements=4,
            element_spacing=0.5,
            reference_frequency=433e6,
        )
        assert len(geom.elements) == 4

        # Should form a 2x2 grid
        x_coords = sorted(set(pos.x for pos in geom.elements))
        y_coords = sorted(set(pos.y for pos in geom.elements))
        assert len(x_coords) == 2
        assert len(y_coords) == 2

    def test_get_wavelength(self):
        """Test wavelength calculation."""
        geom = ArrayGeometry(reference_frequency=300e6)
        expected = SPEED_OF_LIGHT / 300e6
        assert geom.get_wavelength() == pytest.approx(expected)

    def test_get_array_aperture(self):
        """Test array aperture calculation."""
        geom = ArrayGeometry(
            geometry_type="linear",
            num_elements=3,
            element_spacing=0.5,
            reference_frequency=300e6,
        )
        wavelength = SPEED_OF_LIGHT / 300e6
        expected_aperture = 1.0 * wavelength  # 2 spacings for 3 elements
        assert geom.get_array_aperture() == pytest.approx(expected_aperture)

    def test_get_position_matrix(self):
        """Test position matrix generation."""
        geom = ArrayGeometry(
            geometry_type="linear",
            num_elements=2,
            element_spacing=0.5,
            reference_frequency=433e6,
        )
        matrix = geom.get_position_matrix()
        assert matrix.shape == (2, 3)
        assert matrix.dtype == np.float64


class TestArrayConfig:
    """Test ArrayConfig class."""

    def test_initialization_from_geometry(self):
        """Test config creates elements from geometry."""
        geom = ArrayGeometry(
            geometry_type="linear", num_elements=2, reference_frequency=433e6
        )
        config = ArrayConfig(geometry=geom, common_frequency=433e6)

        assert config.num_elements == 2
        assert len(config.elements) == 2
        assert config.common_frequency == 433e6

    def test_enabled_elements(self):
        """Test filtering enabled elements."""
        config = ArrayConfig()
        config.elements = [
            ArrayElement(index=0, enabled=True),
            ArrayElement(index=1, enabled=False),
            ArrayElement(index=2, enabled=True),
        ]
        enabled = config.enabled_elements
        assert len(enabled) == 2
        assert enabled[0].index == 0
        assert enabled[1].index == 2

    def test_get_element_by_index(self):
        """Test element lookup by index."""
        config = create_linear_2_element()
        element = config.get_element_by_index(0)
        assert element is not None
        assert element.index == 0

        missing = config.get_element_by_index(99)
        assert missing is None

    def test_get_calibration_vector(self):
        """Test calibration vector generation."""
        config = create_linear_2_element()
        # Set different calibrations
        config.elements[0].calibration.phase_offset = 0.0
        config.elements[1].calibration.phase_offset = np.pi / 4

        cal_vec = config.get_calibration_vector()
        assert len(cal_vec) == 2
        assert cal_vec.dtype == np.complex64

    def test_wavelength_property(self):
        """Test wavelength property."""
        config = ArrayConfig(common_frequency=300e6)
        expected = SPEED_OF_LIGHT / 300e6
        assert config.wavelength == pytest.approx(expected)

    def test_serialization(self):
        """Test config to dict and back."""
        config = create_linear_2_element(frequency=915e6)
        data = config.to_dict()

        restored = ArrayConfig.from_dict(data)
        assert restored.name == config.name
        assert restored.common_frequency == config.common_frequency
        assert len(restored.elements) == len(config.elements)


class TestSynchronizationConfig:
    """Test SynchronizationConfig class."""

    def test_valid_methods(self):
        """Test valid synchronization methods."""
        for method in ["software", "external_clock", "gpsdo", "correlation"]:
            sync = SynchronizationConfig(method=method)
            assert sync.method == method

    def test_invalid_method(self):
        """Test invalid method raises error."""
        from sdr_module.core.config import ConfigValidationError

        with pytest.raises(ConfigValidationError):
            SynchronizationConfig(method="invalid")

    def test_invalid_time_offset(self):
        """Test invalid time offset raises error."""
        from sdr_module.core.config import ConfigValidationError

        with pytest.raises(ConfigValidationError):
            SynchronizationConfig(max_time_offset_us=-1)

    def test_invalid_correlation_threshold(self):
        """Test invalid correlation threshold raises error."""
        from sdr_module.core.config import ConfigValidationError

        with pytest.raises(ConfigValidationError):
            SynchronizationConfig(correlation_threshold=1.5)


class TestArrayPresets:
    """Test array preset functions."""

    def test_linear_2_element(self):
        """Test 2-element linear preset."""
        config = create_linear_2_element(frequency=433e6)
        assert config.num_elements == 2
        assert config.common_frequency == 433e6
        assert config.geometry.geometry_type == "linear"

    def test_linear_4_element(self):
        """Test 4-element linear preset."""
        config = create_linear_4_element()
        assert config.num_elements == 4

    def test_circular_4_element(self):
        """Test 4-element circular preset."""
        config = create_circular_4_element()
        assert config.num_elements == 4
        assert config.geometry.geometry_type == "circular"

    def test_rectangular_2x2(self):
        """Test 2x2 rectangular preset."""
        config = create_rectangular_2x2()
        assert config.num_elements == 4
        assert config.geometry.geometry_type == "rectangular"

    def test_get_preset(self):
        """Test preset lookup."""
        preset = get_array_preset("linear_2")
        assert preset is not None
        assert preset.num_elements == 2

    def test_list_presets(self):
        """Test preset listing."""
        presets = list_array_presets()
        assert "linear_2" in presets
        assert "linear_4" in presets
        assert "circular_4" in presets
        assert "rectangular_2x2" in presets


class TestTimestampedChunk:
    """Test TimestampedChunk class."""

    def test_initialization(self):
        """Test chunk initialization."""
        samples = np.ones(100, dtype=np.complex64)
        chunk = TimestampedChunk(
            samples=samples,
            timestamp=1000.0,
            sample_index=500,
            sample_rate=2.4e6,
            device_id="rtlsdr_0",
        )
        assert len(chunk.samples) == 100
        assert chunk.timestamp == 1000.0
        assert chunk.sample_index == 500
        assert chunk.sample_rate == 2.4e6
        assert chunk.device_id == "rtlsdr_0"

    def test_duration_property(self):
        """Test duration calculation."""
        samples = np.ones(2400, dtype=np.complex64)
        chunk = TimestampedChunk(
            samples=samples,
            timestamp=0.0,
            sample_index=0,
            sample_rate=2.4e6,
            device_id="test",
        )
        assert chunk.duration == pytest.approx(0.001)  # 1 ms

    def test_end_timestamp(self):
        """Test end timestamp calculation."""
        samples = np.ones(2400, dtype=np.complex64)
        chunk = TimestampedChunk(
            samples=samples,
            timestamp=1.0,
            sample_index=0,
            sample_rate=2.4e6,
            device_id="test",
        )
        assert chunk.end_timestamp == pytest.approx(1.001)

    def test_end_sample_index(self):
        """Test end sample index calculation."""
        samples = np.ones(100, dtype=np.complex64)
        chunk = TimestampedChunk(
            samples=samples,
            timestamp=0.0,
            sample_index=1000,
            sample_rate=2.4e6,
            device_id="test",
        )
        assert chunk.end_sample_index == 1100


class TestTimestampedSampleBuffer:
    """Test TimestampedSampleBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = TimestampedSampleBuffer(
            capacity_chunks=128,
            capacity_samples=1024 * 1024,
            device_id="test_device",
        )
        assert buffer.capacity_chunks == 128
        assert buffer.capacity_samples == 1024 * 1024
        assert buffer.device_id == "test_device"
        assert buffer.available_chunks == 0
        assert buffer.available_samples == 0

    def test_write_read_simple(self):
        """Test simple write and read."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=1000)
        samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)

        success = buffer.write(samples, timestamp=1.0, sample_rate=100.0)
        assert success
        assert buffer.available_chunks == 1
        assert buffer.available_samples == 3

        chunk = buffer.read(timeout=0)
        assert chunk is not None
        assert len(chunk.samples) == 3
        np.testing.assert_array_equal(chunk.samples, samples)
        assert chunk.timestamp == 1.0

    def test_multiple_writes(self):
        """Test multiple chunk writes."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=10000)

        for i in range(5):
            samples = np.ones(100, dtype=np.complex64) * (i + 1)
            buffer.write(samples, timestamp=float(i), sample_rate=100.0)

        assert buffer.available_chunks == 5
        assert buffer.available_samples == 500

    def test_read_samples_spanning_chunks(self):
        """Test reading samples spanning multiple chunks."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=10000)

        # Write two chunks
        samples1 = np.ones(50, dtype=np.complex64)
        samples2 = np.ones(50, dtype=np.complex64) * 2
        buffer.write(samples1, timestamp=1.0, sample_rate=100.0)
        buffer.write(samples2, timestamp=1.5, sample_rate=100.0)

        # Read spanning both
        result = buffer.read_samples(75, timeout=0)
        assert result is not None
        samples, timestamp, index = result
        assert len(samples) == 75
        assert timestamp == 1.0
        assert np.all(samples[:50] == 1.0)
        assert np.all(samples[50:] == 2.0)

    def test_overflow_drop_oldest(self):
        """Test DROP_OLDEST overflow policy."""
        buffer = TimestampedSampleBuffer(
            capacity_chunks=3,
            capacity_samples=1000,
            overflow_policy=BufferOverflowPolicy.DROP_OLDEST,
        )

        # Fill buffer
        for i in range(3):
            buffer.write(
                np.ones(100, dtype=np.complex64) * i,
                timestamp=float(i),
                sample_rate=100.0,
            )

        # Overflow with new chunk
        buffer.write(
            np.ones(100, dtype=np.complex64) * 99,
            timestamp=99.0,
            sample_rate=100.0,
        )

        # Should have dropped oldest
        assert buffer.available_chunks == 3
        chunk = buffer.read(timeout=0)
        assert chunk is not None
        assert chunk.timestamp == 1.0  # Second chunk is now first

    def test_peek(self):
        """Test peek operation."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=1000)
        samples = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
        buffer.write(samples, timestamp=5.0, sample_rate=100.0)

        # Peek shouldn't remove data
        peeked = buffer.peek()
        assert peeked is not None
        np.testing.assert_array_equal(peeked.samples, samples)
        assert buffer.available_chunks == 1

        # Can still read
        chunk = buffer.read(timeout=0)
        np.testing.assert_array_equal(chunk.samples, samples)
        assert buffer.available_chunks == 0

    def test_timestamp_range(self):
        """Test timestamp range tracking."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=10000)

        buffer.write(np.ones(10, dtype=np.complex64), timestamp=1.0, sample_rate=100.0)
        buffer.write(np.ones(10, dtype=np.complex64), timestamp=5.0, sample_rate=100.0)

        first_ts, last_ts = buffer.get_timestamp_range()
        assert first_ts == 1.0
        assert last_ts == 5.0

    def test_phase_offset(self):
        """Test phase offset property."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=1000)
        assert buffer.phase_offset == 0.0

        buffer.phase_offset = np.pi / 4
        assert buffer.phase_offset == pytest.approx(np.pi / 4)

    def test_stats_tracking(self):
        """Test statistics tracking."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=10000)

        samples = np.ones(100, dtype=np.complex64)
        buffer.write(samples, timestamp=1.0, sample_rate=100.0)
        buffer.write(samples, timestamp=2.0, sample_rate=100.0)

        stats = buffer.stats
        assert stats.total_samples_in == 200
        assert stats.total_chunks_in == 2
        assert stats.current_fill == 200

        buffer.read(timeout=0)
        stats = buffer.stats
        assert stats.total_samples_out == 100
        assert stats.total_chunks_out == 1

    def test_clear(self):
        """Test buffer clear."""
        buffer = TimestampedSampleBuffer(capacity_chunks=10, capacity_samples=10000)
        buffer.write(np.ones(100, dtype=np.complex64), timestamp=1.0, sample_rate=100.0)

        assert buffer.available_chunks == 1
        buffer.clear()
        assert buffer.available_chunks == 0
        assert buffer.available_samples == 0


class TestTimestampedBufferThreadSafety:
    """Test thread safety of TimestampedSampleBuffer."""

    def test_concurrent_write_read(self):
        """Test concurrent writes and reads."""
        buffer = TimestampedSampleBuffer(
            capacity_chunks=100, capacity_samples=100000
        )
        write_count = 0
        read_count = 0
        errors = []

        def writer():
            nonlocal write_count
            try:
                for i in range(50):
                    samples = np.random.randn(100) + 1j * np.random.randn(100)
                    samples = samples.astype(np.complex64)
                    buffer.write(samples, timestamp=time.time(), sample_rate=1000.0)
                    write_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            nonlocal read_count
            try:
                for _ in range(50):
                    chunk = buffer.read(timeout=1.0)
                    if chunk is not None:
                        read_count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join(timeout=10)
        reader_thread.join(timeout=10)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert write_count == 50
        assert read_count > 0


class TestAntennaArrayController:
    """Test AntennaArrayController class."""

    def test_initialization(self):
        """Test controller initialization."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        assert controller.num_elements == 2
        assert controller.config == config

    def test_state_property(self):
        """Test state property returns copy."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        state1 = controller.state
        state2 = controller.state

        # Should be different objects (copies)
        assert state1 is not state2
        assert state1.mode == state2.mode

    def test_initial_mode_is_idle(self):
        """Test controller starts in IDLE mode."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        assert controller.state.mode == ArrayOperationMode.IDLE
        assert controller.state.sync_state == SyncState.NOT_SYNCED

    def test_get_buffer_before_init(self):
        """Test buffer access before initialization."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        buffer = controller.get_buffer(0)
        assert buffer is None

    def test_get_device_before_init(self):
        """Test device access before initialization."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        device = controller.get_device(0)
        assert device is None

    def test_context_manager(self):
        """Test context manager protocol."""
        config = create_linear_2_element()

        # This will try to initialize but likely fail without hardware
        # The test verifies the protocol works
        with AntennaArrayController(config) as controller:
            state = controller.state
            assert state.mode == ArrayOperationMode.IDLE

    def test_repr(self):
        """Test string representation."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        repr_str = repr(controller)
        assert "AntennaArrayController" in repr_str
        assert "elements=2" in repr_str


class TestArrayControllerStatus:
    """Test AntennaArrayController status methods."""

    def test_get_status(self):
        """Test comprehensive status retrieval."""
        config = create_linear_2_element()
        controller = AntennaArrayController(config)

        status = controller.get_status()

        assert "mode" in status
        assert "sync_state" in status
        assert "num_elements" in status
        assert "active_elements" in status
        assert "elements" in status
        assert status["mode"] == "idle"
        assert status["num_elements"] == 2


class TestSpeedOfLight:
    """Test speed of light constant."""

    def test_speed_of_light_value(self):
        """Test speed of light is approximately correct."""
        # Standard value: 299,792,458 m/s
        assert SPEED_OF_LIGHT == pytest.approx(299792458.0)

    def test_wavelength_calculation(self):
        """Test wavelength calculation with speed of light."""
        # At 300 MHz, wavelength should be ~1 meter
        freq = 300e6
        wavelength = SPEED_OF_LIGHT / freq
        assert wavelength == pytest.approx(1.0, rel=0.01)

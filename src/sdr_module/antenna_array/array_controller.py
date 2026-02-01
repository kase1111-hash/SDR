"""
Antenna array controller for N-device orchestration.

Provides coordinated control of multiple SDR devices as a phased
antenna array, with synchronized streaming and buffer management.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.device_manager import DeviceManager
from ..core.sample_buffer import BufferOverflowPolicy
from ..devices.base import SDRDevice
from .array_config import ArrayConfig, ArrayElement
from .timestamped_buffer import TimestampedChunk, TimestampedSampleBuffer

logger = logging.getLogger(__name__)


class ArrayOperationMode(Enum):
    """Antenna array operation modes."""

    IDLE = "idle"  # Array not streaming
    RECEIVE = "receive"  # All elements receiving
    CALIBRATION = "calibration"  # Running calibration routine
    BEAMFORMING = "beamforming"  # Active beamforming mode
    DIRECTION_FINDING = "direction_finding"  # DoA estimation mode


class SyncState(Enum):
    """Synchronization state."""

    NOT_SYNCED = "not_synced"
    SYNCING = "syncing"
    SYNCED = "synced"
    SYNC_LOST = "sync_lost"


@dataclass
class ElementState:
    """State of a single array element."""

    index: int
    device_connected: bool = False
    streaming: bool = False
    frequency: float = 0.0
    sample_rate: float = 0.0
    gain: float = 0.0
    buffer_fill_ratio: float = 0.0
    samples_received: int = 0
    last_timestamp: float = 0.0
    phase_offset: float = 0.0  # Estimated phase offset from reference


@dataclass
class ArrayState:
    """Current state of the antenna array system."""

    mode: ArrayOperationMode = ArrayOperationMode.IDLE
    sync_state: SyncState = SyncState.NOT_SYNCED
    num_elements: int = 0
    active_elements: int = 0
    streaming_elements: int = 0
    common_frequency: float = 0.0
    common_sample_rate: float = 0.0
    total_samples_received: int = 0
    last_sync_time: float = 0.0
    element_states: Dict[int, ElementState] = None

    def __post_init__(self) -> None:
        if self.element_states is None:
            self.element_states = {}


class AntennaArrayController:
    """
    Controller for antenna array operation with multiple SDR devices.

    Manages coordinated control of N SDR devices as a coherent
    antenna array, providing:
    - Synchronized streaming from all elements
    - Per-element timestamped buffers
    - Phase alignment and calibration support
    - Unified configuration management

    Thread Safety:
        - State property returns a thread-safe copy
        - All public methods are thread-safe
        - Callbacks execute in device threads
        - Buffer access is thread-safe

    Example:
        config = ArrayConfig(...)
        controller = AntennaArrayController(config)

        with controller:
            controller.start_receive(callback=process_samples)
            # ... do processing ...
            controller.stop_all()
    """

    def __init__(self, config: Optional[ArrayConfig] = None) -> None:
        """
        Initialize antenna array controller.

        Args:
            config: Array configuration, uses default if not provided
        """
        self._config = config or ArrayConfig()
        self._device_manager = DeviceManager()

        # Device tracking
        self._devices: Dict[int, SDRDevice] = {}  # element_index -> device
        self._buffers: Dict[int, TimestampedSampleBuffer] = {}  # element_index -> buffer

        # Thread synchronization
        self._lock = RLock()
        self._stop_event = Event()

        # State
        self._state = ArrayState(
            num_elements=self._config.num_elements,
            common_frequency=self._config.common_frequency,
            common_sample_rate=self._config.common_sample_rate,
        )

        # User callbacks
        self._sample_callback: Optional[
            Callable[[int, np.ndarray, float], None]
        ] = None  # (element_idx, samples, timestamp)
        self._sync_callback: Optional[
            Callable[[Dict[int, TimestampedChunk]], None]
        ] = None  # Synchronized samples from all elements

        # Synchronization tracking
        self._reference_element = self._config.sync.reference_element
        self._phase_offsets: Dict[int, float] = {}
        self._time_offsets: Dict[int, float] = {}

        # Background processing thread
        self._sync_thread: Optional[Thread] = None

    @property
    def config(self) -> ArrayConfig:
        """Get array configuration."""
        return self._config

    @property
    def state(self) -> ArrayState:
        """Get current array state (thread-safe copy)."""
        with self._lock:
            element_states_copy = {
                idx: ElementState(
                    index=es.index,
                    device_connected=es.device_connected,
                    streaming=es.streaming,
                    frequency=es.frequency,
                    sample_rate=es.sample_rate,
                    gain=es.gain,
                    buffer_fill_ratio=es.buffer_fill_ratio,
                    samples_received=es.samples_received,
                    last_timestamp=es.last_timestamp,
                    phase_offset=es.phase_offset,
                )
                for idx, es in self._state.element_states.items()
            }
            return ArrayState(
                mode=self._state.mode,
                sync_state=self._state.sync_state,
                num_elements=self._state.num_elements,
                active_elements=self._state.active_elements,
                streaming_elements=self._state.streaming_elements,
                common_frequency=self._state.common_frequency,
                common_sample_rate=self._state.common_sample_rate,
                total_samples_received=self._state.total_samples_received,
                last_sync_time=self._state.last_sync_time,
                element_states=element_states_copy,
            )

    @property
    def num_elements(self) -> int:
        """Number of configured array elements."""
        return self._config.num_elements

    @property
    def active_elements(self) -> int:
        """Number of connected and active elements."""
        with self._lock:
            return self._state.active_elements

    def get_buffer(self, element_index: int) -> Optional[TimestampedSampleBuffer]:
        """Get the sample buffer for a specific element."""
        with self._lock:
            return self._buffers.get(element_index)

    def get_device(self, element_index: int) -> Optional[SDRDevice]:
        """Get the device for a specific element."""
        with self._lock:
            return self._devices.get(element_index)

    def initialize(self) -> bool:
        """
        Initialize and connect to all configured SDR devices.

        Returns:
            True if at least one device connected successfully
        """
        logger.info(
            f"Initializing antenna array with {self._config.num_elements} elements..."
        )

        # Scan for devices
        self._device_manager.scan_devices()

        success_count = 0

        for element in self._config.enabled_elements:
            element_idx = element.index

            # Create buffer for this element
            buffer = TimestampedSampleBuffer(
                capacity_chunks=self._config.buffer_capacity_chunks,
                capacity_samples=self._config.buffer_capacity_samples,
                overflow_policy=BufferOverflowPolicy.DROP_OLDEST,
                device_id=f"{element.device_type}_{element.device_index}",
            )
            self._buffers[element_idx] = buffer

            # Get device configuration
            device_config = self._config.get_device_config(element_idx)

            # Open device
            device = self._device_manager.open_device(
                element.device_type, element.device_index, device_config
            )

            if device:
                self._devices[element_idx] = device
                success_count += 1

                # Initialize element state
                with self._lock:
                    self._state.element_states[element_idx] = ElementState(
                        index=element_idx,
                        device_connected=True,
                        frequency=device.state.frequency,
                        sample_rate=device.state.sample_rate,
                        gain=device.state.gain,
                    )

                logger.info(
                    f"Element {element_idx}: Connected {element.device_type}[{element.device_index}]"
                )
            else:
                logger.warning(
                    f"Element {element_idx}: Failed to connect {element.device_type}[{element.device_index}]"
                )
                with self._lock:
                    self._state.element_states[element_idx] = ElementState(
                        index=element_idx, device_connected=False
                    )

        with self._lock:
            self._state.active_elements = success_count

        if success_count > 0:
            logger.info(
                f"Antenna array initialized: {success_count}/{self._config.num_elements} elements connected"
            )
        else:
            logger.error("No array elements could be connected")

        return success_count > 0

    def shutdown(self) -> None:
        """Shutdown and disconnect all devices."""
        logger.info("Shutting down antenna array...")

        self.stop_all()
        self._device_manager.close_all()

        with self._lock:
            self._devices.clear()
            self._buffers.clear()
            self._state.active_elements = 0
            self._state.streaming_elements = 0
            self._state.mode = ArrayOperationMode.IDLE

        logger.info("Antenna array shutdown complete")

    def set_frequency(self, freq_hz: float) -> Tuple[int, int]:
        """
        Set center frequency for all elements.

        Args:
            freq_hz: Center frequency in Hz

        Returns:
            Tuple of (success_count, total_count)
        """
        success = 0
        total = 0

        with self._lock:
            for element_idx, device in self._devices.items():
                total += 1
                if device.set_frequency(freq_hz):
                    success += 1
                    if element_idx in self._state.element_states:
                        self._state.element_states[element_idx].frequency = freq_hz

            self._state.common_frequency = freq_hz

        logger.info(f"Set frequency to {freq_hz/1e6:.3f} MHz: {success}/{total} elements")
        return (success, total)

    def set_sample_rate(self, rate_hz: float) -> Tuple[int, int]:
        """
        Set sample rate for all elements.

        Args:
            rate_hz: Sample rate in Hz

        Returns:
            Tuple of (success_count, total_count)
        """
        success = 0
        total = 0

        with self._lock:
            for element_idx, device in self._devices.items():
                total += 1
                if device.set_sample_rate(rate_hz):
                    success += 1
                    if element_idx in self._state.element_states:
                        self._state.element_states[element_idx].sample_rate = rate_hz

            self._state.common_sample_rate = rate_hz

        logger.info(f"Set sample rate to {rate_hz/1e6:.3f} MS/s: {success}/{total} elements")
        return (success, total)

    def set_gain(self, gain_db: float) -> Tuple[int, int]:
        """
        Set gain for all elements.

        Args:
            gain_db: Gain in dB

        Returns:
            Tuple of (success_count, total_count)
        """
        success = 0
        total = 0

        with self._lock:
            for element_idx, device in self._devices.items():
                total += 1
                if device.set_gain(gain_db):
                    success += 1
                    if element_idx in self._state.element_states:
                        self._state.element_states[element_idx].gain = gain_db

        logger.info(f"Set gain to {gain_db:.1f} dB: {success}/{total} elements")
        return (success, total)

    def set_element_frequency(self, element_index: int, freq_hz: float) -> bool:
        """Set frequency for a specific element."""
        with self._lock:
            device = self._devices.get(element_index)
            if device is None:
                return False
            if device.set_frequency(freq_hz):
                if element_index in self._state.element_states:
                    self._state.element_states[element_index].frequency = freq_hz
                return True
            return False

    def set_element_gain(self, element_index: int, gain_db: float) -> bool:
        """Set gain for a specific element."""
        with self._lock:
            device = self._devices.get(element_index)
            if device is None:
                return False
            if device.set_gain(gain_db):
                if element_index in self._state.element_states:
                    self._state.element_states[element_index].gain = gain_db
                return True
            return False

    def start_receive(
        self,
        sample_callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
        sync_callback: Optional[Callable[[Dict[int, TimestampedChunk]], None]] = None,
    ) -> bool:
        """
        Start receiving on all connected elements.

        Args:
            sample_callback: Called for each element's samples:
                            callback(element_index, samples, timestamp)
            sync_callback: Called with synchronized samples from all elements:
                          callback(dict of element_index -> TimestampedChunk)

        Returns:
            True if at least one element started receiving
        """
        with self._lock:
            if self._state.mode != ArrayOperationMode.IDLE:
                logger.warning("Array is already in active mode")
                return False

            self._sample_callback = sample_callback
            self._sync_callback = sync_callback
            self._stop_event.clear()

        success_count = 0

        for element_idx, device in self._devices.items():
            buffer = self._buffers.get(element_idx)
            if buffer is None:
                continue

            # Capture references for closure
            elem_idx = element_idx
            elem_buffer = buffer
            sample_rate = device.state.sample_rate

            # Capture callback reference for thread-safe access
            with self._lock:
                user_callback = self._sample_callback

            def make_callback(idx: int, buf: TimestampedSampleBuffer, rate: float, user_cb: Optional[Callable]):
                def rx_callback(samples: np.ndarray) -> None:
                    if self._stop_event.is_set():
                        return

                    timestamp = time.time()

                    # Write to timestamped buffer
                    buf.write(samples, timestamp=timestamp, sample_rate=rate)

                    # Update state
                    with self._lock:
                        if idx in self._state.element_states:
                            es = self._state.element_states[idx]
                            es.samples_received += len(samples)
                            es.last_timestamp = timestamp
                            es.buffer_fill_ratio = buf.stats.fill_ratio
                        self._state.total_samples_received += len(samples)

                    # User callback
                    if user_cb:
                        user_cb(idx, samples, timestamp)

                return rx_callback

            callback = make_callback(elem_idx, elem_buffer, sample_rate, user_callback)

            if device.start_rx(callback):
                success_count += 1
                with self._lock:
                    if elem_idx in self._state.element_states:
                        self._state.element_states[elem_idx].streaming = True
                logger.info(f"Element {elem_idx}: Started receiving")
            else:
                logger.warning(f"Element {elem_idx}: Failed to start receiving")

        with self._lock:
            self._state.streaming_elements = success_count
            if success_count > 0:
                self._state.mode = ArrayOperationMode.RECEIVE

        # Start synchronization thread if sync callback provided
        if sync_callback and success_count > 1:
            self._start_sync_thread()

        if success_count > 0:
            logger.info(f"Array receiving: {success_count} elements active")
        else:
            logger.error("Failed to start any array elements")

        return success_count > 0

    def stop_all(self) -> None:
        """Stop all streaming."""
        self._stop_event.set()

        # Stop sync thread
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
            self._sync_thread = None

        with self._lock:
            for element_idx, device in self._devices.items():
                if device.state.is_streaming:
                    device.stop_rx()
                    if element_idx in self._state.element_states:
                        self._state.element_states[element_idx].streaming = False
                    logger.info(f"Element {element_idx}: Stopped receiving")

            self._state.streaming_elements = 0
            self._state.mode = ArrayOperationMode.IDLE
            self._sample_callback = None
            self._sync_callback = None

    def stop_element(self, element_index: int) -> bool:
        """Stop a specific element."""
        with self._lock:
            device = self._devices.get(element_index)
            if device is None:
                return False

            if device.state.is_streaming:
                device.stop_rx()
                if element_index in self._state.element_states:
                    self._state.element_states[element_index].streaming = False
                self._state.streaming_elements -= 1
                return True
            return False

    def read_samples(
        self, element_index: int, n_samples: int, timeout: float = 1.0
    ) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Read samples from a specific element's buffer.

        Args:
            element_index: Element to read from
            n_samples: Number of samples to read
            timeout: Timeout in seconds

        Returns:
            Tuple of (samples, timestamp, sample_index) or None
        """
        buffer = self._buffers.get(element_index)
        if buffer is None:
            return None
        return buffer.read_samples(n_samples, timeout)

    def read_chunk(
        self, element_index: int, timeout: float = 1.0
    ) -> Optional[TimestampedChunk]:
        """
        Read a chunk from a specific element's buffer.

        Args:
            element_index: Element to read from
            timeout: Timeout in seconds

        Returns:
            TimestampedChunk or None
        """
        buffer = self._buffers.get(element_index)
        if buffer is None:
            return None
        return buffer.read(timeout)

    def read_all_chunks(
        self, timeout: float = 1.0
    ) -> Dict[int, Optional[TimestampedChunk]]:
        """
        Read one chunk from each element's buffer.

        Args:
            timeout: Timeout in seconds

        Returns:
            Dict mapping element_index to TimestampedChunk (or None if unavailable)
        """
        chunks = {}
        for element_idx, buffer in self._buffers.items():
            chunks[element_idx] = buffer.read(timeout=timeout)
        return chunks

    def get_synchronized_samples(
        self, n_samples: int, timeout: float = 1.0
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        Get approximately time-aligned samples from all elements.

        Note: Without hardware synchronization, this provides best-effort
        alignment based on timestamps. True phase coherence requires
        external clock synchronization or calibration.

        Args:
            n_samples: Number of samples to read from each element
            timeout: Timeout in seconds

        Returns:
            Dict mapping element_index to samples, or None if insufficient data
        """
        result = {}
        timestamps = {}

        for element_idx, buffer in self._buffers.items():
            data = buffer.read_samples(n_samples, timeout)
            if data is None:
                return None
            samples, timestamp, _ = data
            result[element_idx] = samples
            timestamps[element_idx] = timestamp

        # Log timing spread
        if timestamps:
            min_ts = min(timestamps.values())
            max_ts = max(timestamps.values())
            spread_us = (max_ts - min_ts) * 1e6
            if spread_us > self._config.sync.max_time_offset_us:
                logger.warning(
                    f"Timestamp spread {spread_us:.1f} us exceeds threshold "
                    f"{self._config.sync.max_time_offset_us:.1f} us"
                )

        return result

    def apply_calibration(self, samples: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Apply calibration corrections to samples from all elements.

        Args:
            samples: Dict mapping element_index to samples

        Returns:
            Dict with calibration-corrected samples
        """
        corrected = {}

        for element_idx, element_samples in samples.items():
            element = self._config.get_element_by_index(element_idx)
            if element is None:
                corrected[element_idx] = element_samples
                continue

            # Apply phase and amplitude correction
            correction = element.calibration.get_correction_phasor()
            corrected[element_idx] = element_samples * correction

        return corrected

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of antenna array system."""
        with self._lock:
            elements_status = {}
            for element_idx, es in self._state.element_states.items():
                buffer = self._buffers.get(element_idx)
                buffer_stats = buffer.stats if buffer else None

                elements_status[element_idx] = {
                    "connected": es.device_connected,
                    "streaming": es.streaming,
                    "frequency_mhz": es.frequency / 1e6,
                    "sample_rate_msps": es.sample_rate / 1e6,
                    "gain_db": es.gain,
                    "samples_received": es.samples_received,
                    "buffer_fill_ratio": (
                        buffer_stats.fill_ratio if buffer_stats else 0.0
                    ),
                    "phase_offset_rad": es.phase_offset,
                }

            return {
                "mode": self._state.mode.value,
                "sync_state": self._state.sync_state.value,
                "num_elements": self._state.num_elements,
                "active_elements": self._state.active_elements,
                "streaming_elements": self._state.streaming_elements,
                "common_frequency_mhz": self._state.common_frequency / 1e6,
                "common_sample_rate_msps": self._state.common_sample_rate / 1e6,
                "total_samples_received": self._state.total_samples_received,
                "elements": elements_status,
            }

    def _start_sync_thread(self) -> None:
        """Start background thread for synchronized sample delivery."""
        if self._sync_thread and self._sync_thread.is_alive():
            return

        self._sync_thread = Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def _sync_loop(self) -> None:
        """Background thread that delivers synchronized samples."""
        logger.info("Synchronization thread started")

        while not self._stop_event.is_set():
            try:
                # Collect one chunk from each element
                chunks = {}
                all_available = True

                for element_idx, buffer in self._buffers.items():
                    if buffer.available_chunks > 0:
                        chunk = buffer.read(timeout=0.1)
                        if chunk:
                            chunks[element_idx] = chunk
                    else:
                        all_available = False

                # Only deliver if we have data from all streaming elements
                if chunks and all_available:
                    with self._lock:
                        callback = self._sync_callback
                        self._state.last_sync_time = time.time()

                    if callback:
                        callback(chunks)
                else:
                    # Wait a bit before checking again
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(0.01)

        logger.info("Synchronization thread stopped")

    def __enter__(self) -> "AntennaArrayController":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> bool:
        """Context manager exit."""
        self.shutdown()
        return False

    def __repr__(self) -> str:
        return (
            f"<AntennaArrayController elements={self.num_elements} "
            f"active={self.active_elements} mode={self._state.mode.value}>"
        )

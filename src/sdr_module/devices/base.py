"""
Base SDR device abstraction layer.

Provides a unified interface for all SDR hardware devices.
"""

import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Event, RLock, Thread
from typing import Callable, List, Optional

import numpy as np


class DeviceCapability(Enum):
    """SDR device capabilities."""

    RX = auto()  # Receive capability
    TX = auto()  # Transmit capability
    FULL_DUPLEX = auto()  # Simultaneous TX/RX
    HALF_DUPLEX = auto()  # TX or RX, not both
    BIAS_TEE = auto()  # Bias tee for active antennas
    EXT_CLOCK = auto()  # External clock reference
    DIRECT_SAMPLE = auto()  # Direct sampling mode (HF)


@dataclass
class DeviceInfo:
    """SDR device information."""

    name: str
    serial: str
    manufacturer: str
    product: str
    index: int = 0
    capabilities: List[DeviceCapability] = field(default_factory=list)


@dataclass
class DeviceSpec:
    """SDR device specifications."""

    freq_min: float  # Minimum frequency in Hz
    freq_max: float  # Maximum frequency in Hz
    sample_rate_min: float  # Minimum sample rate in Hz
    sample_rate_max: float  # Maximum sample rate in Hz
    bandwidth_max: float  # Maximum instantaneous bandwidth in Hz
    adc_bits: int  # ADC resolution in bits
    gain_min: float  # Minimum gain in dB
    gain_max: float  # Maximum gain in dB
    max_input_power: float  # Maximum input power in dBm
    tx_power_min: Optional[float] = None  # Min TX power in dBm
    tx_power_max: Optional[float] = None  # Max TX power in dBm


@dataclass
class DeviceState:
    """Current SDR device state."""

    frequency: float = 0.0  # Center frequency in Hz
    sample_rate: float = 0.0  # Sample rate in Hz
    bandwidth: float = 0.0  # Filter bandwidth in Hz
    gain: float = 0.0  # Current gain in dB
    gain_mode: str = "manual"  # "auto" or "manual"
    is_streaming: bool = False
    is_transmitting: bool = False
    bias_tee_enabled: bool = False
    amp_enabled: bool = False


class SDRDevice(ABC):
    """
    Abstract base class for SDR devices.

    Provides a unified interface for RTL-SDR, HackRF, and other SDR hardware.
    """

    def __init__(self):
        self._info: Optional[DeviceInfo] = None
        self._spec: Optional[DeviceSpec] = None
        self._state: DeviceState = DeviceState()
        self._state_lock: RLock = RLock()  # Protects _state modifications
        self._is_open: bool = False
        self._rx_callback: Optional[Callable[[np.ndarray], None]] = None
        self._rx_thread: Optional[Thread] = None
        self._stop_event: Event = Event()
        self._sample_queue: queue.Queue = queue.Queue(maxsize=100)

    @property
    def info(self) -> Optional[DeviceInfo]:
        """Get device information."""
        return self._info

    @property
    def spec(self) -> Optional[DeviceSpec]:
        """Get device specifications."""
        return self._spec

    @property
    def state(self) -> DeviceState:
        """Get current device state (thread-safe copy)."""
        with self._state_lock:
            return DeviceState(
                frequency=self._state.frequency,
                sample_rate=self._state.sample_rate,
                bandwidth=self._state.bandwidth,
                gain=self._state.gain,
                gain_mode=self._state.gain_mode,
                is_streaming=self._state.is_streaming,
                is_transmitting=self._state.is_transmitting,
                bias_tee_enabled=self._state.bias_tee_enabled,
                amp_enabled=self._state.amp_enabled,
            )

    @property
    def is_open(self) -> bool:
        """Check if device is open."""
        return self._is_open

    @abstractmethod
    def open(self, index: int = 0) -> bool:
        """
        Open the SDR device.

        Args:
            index: Device index if multiple devices present

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the SDR device and release resources."""
        pass

    @abstractmethod
    def set_frequency(self, freq_hz: float) -> bool:
        """
        Set the center frequency.

        Args:
            freq_hz: Center frequency in Hz

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def set_sample_rate(self, rate_hz: float) -> bool:
        """
        Set the sample rate.

        Args:
            rate_hz: Sample rate in Hz

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def set_bandwidth(self, bw_hz: float) -> bool:
        """
        Set the filter bandwidth.

        Args:
            bw_hz: Bandwidth in Hz

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def set_gain(self, gain_db: float) -> bool:
        """
        Set the gain.

        Args:
            gain_db: Gain in dB

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def set_gain_mode(self, auto: bool) -> bool:
        """
        Set gain mode to auto or manual.

        Args:
            auto: True for automatic gain control, False for manual

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def start_rx(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """
        Start receiving samples.

        Args:
            callback: Optional callback function for received samples

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def stop_rx(self) -> bool:
        """
        Stop receiving samples.

        Returns:
            True if successful, False otherwise
        """
        pass

    def read_samples(
        self, num_samples: int, timeout: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Read samples from the device.

        Args:
            num_samples: Number of complex samples to read
            timeout: Timeout in seconds

        Returns:
            Complex numpy array of samples, or None on error
        """
        try:
            return self._sample_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_capability(self, cap: DeviceCapability) -> bool:
        """Check if device has a specific capability."""
        if self._info is None:
            return False
        return cap in self._info.capabilities

    def can_transmit(self) -> bool:
        """Check if device can transmit."""
        return self.has_capability(DeviceCapability.TX)

    def can_receive(self) -> bool:
        """Check if device can receive."""
        return self.has_capability(DeviceCapability.RX)

    # TX methods (optional, not all devices support TX)
    def start_tx(self, callback: Optional[Callable[[], np.ndarray]] = None) -> bool:
        """Start transmitting samples."""
        raise NotImplementedError("This device does not support transmit")

    def stop_tx(self) -> bool:
        """Stop transmitting samples."""
        raise NotImplementedError("This device does not support transmit")

    def write_samples(self, samples: np.ndarray) -> bool:
        """Write samples to transmit."""
        raise NotImplementedError("This device does not support transmit")

    def set_tx_gain(self, gain_db: float) -> bool:
        """Set transmit gain."""
        raise NotImplementedError("This device does not support transmit")

    # Bias tee control (optional)
    def set_bias_tee(self, enabled: bool) -> bool:
        """Enable or disable bias tee."""
        raise NotImplementedError("This device does not support bias tee")

    # Amplifier control (optional)
    def set_amp(self, enabled: bool) -> bool:
        """Enable or disable RF amplifier."""
        raise NotImplementedError("This device does not support amplifier control")

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        if self._info:
            return f"<{self.__class__.__name__} {self._info.name} @ {self._state.frequency/1e6:.3f} MHz>"
        return f"<{self.__class__.__name__} (not opened)>"

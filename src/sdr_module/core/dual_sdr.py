"""
Dual-SDR controller for simultaneous RTL-SDR and HackRF operation.

Provides coordinated control of both devices for:
- Dual RX monitoring
- Full-duplex operation (RTL-SDR RX + HackRF TX)
- TX monitoring
- Wideband scanning
"""

import logging
from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..devices.hackrf import HackRFDevice
from ..devices.rtlsdr import RTLSDRDevice
from .config import SDRConfig
from .device_manager import DeviceManager
from .sample_buffer import SampleBuffer

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Dual-SDR operation modes."""

    DUAL_RX = "dual_rx"  # Both devices receiving
    FULL_DUPLEX = "full_duplex"  # RTL-SDR RX + HackRF TX
    TX_MONITOR = "tx_monitor"  # Monitor own transmission
    WIDEBAND_SCAN = "wideband_scan"  # Coordinated scanning
    RELAY = "relay"  # Receive and retransmit


@dataclass
class DualSDRState:
    """Current state of dual-SDR system."""

    mode: OperationMode = OperationMode.DUAL_RX
    rtlsdr_streaming: bool = False
    hackrf_streaming: bool = False
    hackrf_transmitting: bool = False
    rtlsdr_frequency: float = 0.0
    hackrf_frequency: float = 0.0
    is_synchronized: bool = False


class DualSDRController:
    """
    Controller for dual-SDR operation.

    Manages coordinated operation of RTL-SDR and HackRF One
    for various use cases including dual receive, full-duplex,
    and transmit monitoring.
    """

    def __init__(self, config: Optional[SDRConfig] = None):
        """
        Initialize dual-SDR controller.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self._config = config or SDRConfig()
        self._device_manager = DeviceManager()
        self._rtlsdr: Optional[RTLSDRDevice] = None
        self._hackrf: Optional[HackRFDevice] = None

        # Sample buffers
        self._rtlsdr_buffer = SampleBuffer(capacity=2 * 1024 * 1024)
        self._hackrf_buffer = SampleBuffer(capacity=8 * 1024 * 1024)

        # Thread synchronization - protects _state and callback references
        self._lock = RLock()

        # State
        self._state = DualSDRState()
        self._stop_event = Event()

        # Callbacks
        self._rtlsdr_callback: Optional[Callable[[np.ndarray], None]] = None
        self._hackrf_callback: Optional[Callable[[np.ndarray], None]] = None
        self._tx_generator: Optional[Callable[[], np.ndarray]] = None

    @property
    def state(self) -> DualSDRState:
        """Get current controller state (thread-safe copy)."""
        with self._lock:
            return DualSDRState(
                mode=self._state.mode,
                rtlsdr_streaming=self._state.rtlsdr_streaming,
                hackrf_streaming=self._state.hackrf_streaming,
                hackrf_transmitting=self._state.hackrf_transmitting,
                rtlsdr_frequency=self._state.rtlsdr_frequency,
                hackrf_frequency=self._state.hackrf_frequency,
                is_synchronized=self._state.is_synchronized,
            )

    @property
    def rtlsdr(self) -> Optional[RTLSDRDevice]:
        """Get RTL-SDR device."""
        return self._rtlsdr

    @property
    def hackrf(self) -> Optional[HackRFDevice]:
        """Get HackRF device."""
        return self._hackrf

    @property
    def rtlsdr_buffer(self) -> SampleBuffer:
        """Get RTL-SDR sample buffer."""
        return self._rtlsdr_buffer

    @property
    def hackrf_buffer(self) -> SampleBuffer:
        """Get HackRF sample buffer."""
        return self._hackrf_buffer

    def initialize(self) -> bool:
        """
        Initialize and connect to both SDR devices.

        Returns:
            True if both devices connected successfully
        """
        logger.info("Initializing dual-SDR system...")

        # Scan for devices
        self._device_manager.scan_devices()

        if not self._device_manager.has_dual_sdr():
            logger.warning("Dual-SDR not available, checking individual devices...")

        # Try to connect RTL-SDR
        self._rtlsdr = self._device_manager.get_rtlsdr()
        if self._rtlsdr:
            self._device_manager.apply_config(
                self._rtlsdr, self._config.dual_sdr.rtlsdr
            )
            logger.info("RTL-SDR initialized")
        else:
            logger.warning("RTL-SDR not available")

        # Try to connect HackRF
        self._hackrf = self._device_manager.get_hackrf()
        if self._hackrf:
            self._device_manager.apply_config(
                self._hackrf, self._config.dual_sdr.hackrf
            )
            logger.info("HackRF initialized")
        else:
            logger.warning("HackRF not available")

        # Update state
        if self._rtlsdr:
            self._state.rtlsdr_frequency = self._rtlsdr.state.frequency
        if self._hackrf:
            self._state.hackrf_frequency = self._hackrf.state.frequency

        success = self._rtlsdr is not None or self._hackrf is not None
        if success:
            logger.info("Dual-SDR system initialized")
        else:
            logger.error("No SDR devices available")

        return success

    def shutdown(self) -> None:
        """Shutdown and disconnect all devices."""
        logger.info("Shutting down dual-SDR system...")

        self.stop_all()
        self._device_manager.close_all()

        self._rtlsdr = None
        self._hackrf = None

        logger.info("Dual-SDR system shutdown complete")

    def set_mode(self, mode: OperationMode) -> bool:
        """
        Set the operation mode.

        Args:
            mode: Desired operation mode

        Returns:
            True if mode change successful
        """
        with self._lock:
            if self._state.rtlsdr_streaming or self._state.hackrf_streaming:
                logger.warning("Stop streaming before changing mode")
                return False

            # Validate mode requirements
            if mode == OperationMode.FULL_DUPLEX:
                if not self._rtlsdr or not self._hackrf:
                    logger.error("Full-duplex requires both RTL-SDR and HackRF")
                    return False

            if mode == OperationMode.TX_MONITOR:
                if not self._hackrf:
                    logger.error("TX monitor requires HackRF")
                    return False

            self._state.mode = mode
            logger.info(f"Operation mode set to: {mode.value}")
            return True

    def set_rtlsdr_frequency(self, freq_hz: float) -> bool:
        """Set RTL-SDR center frequency."""
        with self._lock:
            if self._rtlsdr is None:
                return False
            if self._rtlsdr.set_frequency(freq_hz):
                self._state.rtlsdr_frequency = freq_hz
                return True
            return False

    def set_hackrf_frequency(self, freq_hz: float) -> bool:
        """Set HackRF center frequency."""
        with self._lock:
            if self._hackrf is None:
                return False
            if self._hackrf.set_frequency(freq_hz):
                self._state.hackrf_frequency = freq_hz
                return True
            return False

    def set_frequencies(self, rtlsdr_hz: float, hackrf_hz: float) -> Tuple[bool, bool]:
        """Set both frequencies at once."""
        return (
            self.set_rtlsdr_frequency(rtlsdr_hz),
            self.set_hackrf_frequency(hackrf_hz),
        )

    def start_dual_rx(
        self,
        rtlsdr_callback: Optional[Callable[[np.ndarray], None]] = None,
        hackrf_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> bool:
        """
        Start dual receive mode.

        Both RTL-SDR and HackRF receive simultaneously on their
        configured frequencies.

        Args:
            rtlsdr_callback: Callback for RTL-SDR samples
            hackrf_callback: Callback for HackRF samples

        Returns:
            True if both started successfully
        """
        with self._lock:
            if self._state.mode != OperationMode.DUAL_RX:
                self.set_mode(OperationMode.DUAL_RX)

            self._rtlsdr_callback = rtlsdr_callback
            self._hackrf_callback = hackrf_callback
            self._stop_event.clear()

            # Capture callback references for thread-safe access in closures
            rtl_user_cb = self._rtlsdr_callback
            hackrf_user_cb = self._hackrf_callback

        success = True

        # Start RTL-SDR
        if self._rtlsdr:

            def rtl_cb(samples):
                self._rtlsdr_buffer.write(samples)
                if rtl_user_cb:
                    rtl_user_cb(samples)

            if self._rtlsdr.start_rx(rtl_cb):
                with self._lock:
                    self._state.rtlsdr_streaming = True
                logger.info("RTL-SDR RX started")
            else:
                success = False

        # Start HackRF
        if self._hackrf:

            def hackrf_cb(samples):
                self._hackrf_buffer.write(samples)
                if hackrf_user_cb:
                    hackrf_user_cb(samples)

            if self._hackrf.start_rx(hackrf_cb):
                with self._lock:
                    self._state.hackrf_streaming = True
                logger.info("HackRF RX started")
            else:
                success = False

        # Clean up callbacks for devices that didn't start
        with self._lock:
            if not self._state.rtlsdr_streaming:
                self._rtlsdr_callback = None
            if not self._state.hackrf_streaming:
                self._hackrf_callback = None

        return success

    def start_full_duplex(
        self,
        rx_callback: Optional[Callable[[np.ndarray], None]] = None,
        tx_generator: Optional[Callable[[], np.ndarray]] = None,
    ) -> bool:
        """
        Start full-duplex mode.

        RTL-SDR receives while HackRF transmits.

        Args:
            rx_callback: Callback for RTL-SDR received samples
            tx_generator: Generator function for TX samples

        Returns:
            True if started successfully
        """
        with self._lock:
            if not self._rtlsdr or not self._hackrf:
                logger.error("Full-duplex requires both devices")
                return False

            if self._state.mode != OperationMode.FULL_DUPLEX:
                self.set_mode(OperationMode.FULL_DUPLEX)

            self._rtlsdr_callback = rx_callback
            self._tx_generator = tx_generator
            self._stop_event.clear()

            # Capture callback reference for thread-safe access in closure
            rx_user_cb = self._rtlsdr_callback

        # Start RTL-SDR RX
        def rtl_cb(samples):
            self._rtlsdr_buffer.write(samples)
            if rx_user_cb:
                rx_user_cb(samples)

        if not self._rtlsdr.start_rx(rtl_cb):
            logger.error("Failed to start RTL-SDR RX")
            # Clean up callbacks on failure
            with self._lock:
                self._rtlsdr_callback = None
                self._tx_generator = None
            return False

        with self._lock:
            self._state.rtlsdr_streaming = True
        logger.info("RTL-SDR RX started (full-duplex)")

        # Start HackRF TX
        if self._tx_generator:
            if not self._hackrf.start_tx(self._tx_generator):
                logger.error("Failed to start HackRF TX")
                self._rtlsdr.stop_rx()
                with self._lock:
                    self._state.rtlsdr_streaming = False
                    self._rtlsdr_callback = None  # Clean up callback on failure
                    self._tx_generator = None
                return False

            with self._lock:
                self._state.hackrf_transmitting = True
            logger.info("HackRF TX started (full-duplex)")

        return True

    def start_tx_monitor(
        self,
        tx_generator: Callable[[], np.ndarray],
        monitor_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> bool:
        """
        Start TX monitoring mode.

        HackRF transmits while RTL-SDR monitors the transmission.
        RTL-SDR should be tuned to the same frequency (with attenuation).

        Args:
            tx_generator: Generator function for TX samples
            monitor_callback: Callback for monitored samples

        Returns:
            True if started successfully
        """
        if not self._hackrf:
            logger.error("TX monitor requires HackRF")
            return False

        if self._state.mode != OperationMode.TX_MONITOR:
            self.set_mode(OperationMode.TX_MONITOR)

        # Tune RTL-SDR to same frequency as HackRF (user should add attenuation!)
        if self._rtlsdr:
            logger.warning("Ensure RTL-SDR input is properly attenuated!")
            self.set_rtlsdr_frequency(self._state.hackrf_frequency)

        return self.start_full_duplex(monitor_callback, tx_generator)

    def stop_all(self) -> None:
        """Stop all streaming and transmission."""
        self._stop_event.set()

        with self._lock:
            if self._rtlsdr and self._state.rtlsdr_streaming:
                self._rtlsdr.stop_rx()
                self._state.rtlsdr_streaming = False
                self._rtlsdr_callback = None
                logger.info("RTL-SDR RX stopped")

            if self._hackrf:
                if self._state.hackrf_streaming:
                    self._hackrf.stop_rx()
                    self._state.hackrf_streaming = False
                    self._hackrf_callback = None
                    logger.info("HackRF RX stopped")

                if self._state.hackrf_transmitting:
                    self._hackrf.stop_tx()
                    self._state.hackrf_transmitting = False
                    self._tx_generator = None
                    logger.info("HackRF TX stopped")

    def stop_rtlsdr(self) -> None:
        """Stop RTL-SDR streaming only."""
        with self._lock:
            if self._rtlsdr and self._state.rtlsdr_streaming:
                self._rtlsdr.stop_rx()
                self._state.rtlsdr_streaming = False
                self._rtlsdr_callback = None

    def stop_hackrf(self) -> None:
        """Stop HackRF streaming/transmission only."""
        with self._lock:
            if self._hackrf:
                if self._state.hackrf_streaming:
                    self._hackrf.stop_rx()
                    self._state.hackrf_streaming = False
                    self._hackrf_callback = None
                if self._state.hackrf_transmitting:
                    self._hackrf.stop_tx()
                    self._state.hackrf_transmitting = False
                    self._tx_generator = None

    def read_rtlsdr_samples(
        self, n_samples: int, timeout: float = 1.0
    ) -> Optional[np.ndarray]:
        """Read samples from RTL-SDR buffer."""
        return self._rtlsdr_buffer.read(n_samples, timeout)

    def read_hackrf_samples(
        self, n_samples: int, timeout: float = 1.0
    ) -> Optional[np.ndarray]:
        """Read samples from HackRF buffer."""
        return self._hackrf_buffer.read(n_samples, timeout)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of dual-SDR system."""
        with self._lock:
            rtlsdr_status: Dict[str, Any] = {
                "connected": self._rtlsdr is not None,
                "streaming": self._state.rtlsdr_streaming,
                "frequency_mhz": self._state.rtlsdr_frequency / 1e6,
                "buffer_fill": self._rtlsdr_buffer.stats.fill_ratio,
            }
            hackrf_status: Dict[str, Any] = {
                "connected": self._hackrf is not None,
                "streaming": self._state.hackrf_streaming,
                "transmitting": self._state.hackrf_transmitting,
                "frequency_mhz": self._state.hackrf_frequency / 1e6,
                "buffer_fill": self._hackrf_buffer.stats.fill_ratio,
            }

            # Add device-specific info
            if self._rtlsdr:
                rtlsdr_status["sample_rate"] = self._rtlsdr.state.sample_rate
                rtlsdr_status["gain"] = self._rtlsdr.state.gain

            if self._hackrf:
                hackrf_status["sample_rate"] = self._hackrf.state.sample_rate
                hackrf_status["gain"] = self._hackrf.state.gain

            return {
                "mode": self._state.mode.value,
                "rtlsdr": rtlsdr_status,
                "hackrf": hackrf_status,
            }

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False

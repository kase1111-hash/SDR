"""
Device manager for SDR hardware detection and management.

Handles device enumeration, connection management, and
provides a unified interface for accessing SDR devices.
"""

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Optional, Type, cast

from ..devices.base import DeviceInfo, SDRDevice
from ..devices.hackrf import HackRFDevice
from ..devices.mxk2_keyer import MXK2Keyer
from ..devices.rtlsdr import RTLSDRDevice
from .config import DeviceConfig, KeyerConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectedDevice:
    """Information about a detected SDR device."""

    info: DeviceInfo
    device_class: Type[SDRDevice]
    is_available: bool = True


class DeviceManager:
    """
    Manages SDR device detection, connection, and lifecycle.

    Provides a central point for:
    - Enumerating available SDR devices
    - Creating device instances
    - Managing device connections
    - Applying configurations
    """

    # Registered device types
    DEVICE_TYPES: Dict[str, Type[SDRDevice]] = {
        "rtlsdr": RTLSDRDevice,
        "hackrf": HackRFDevice,
        "mxk2_keyer": MXK2Keyer,
    }

    def __init__(self):
        self._lock = RLock()  # Protects _devices dictionary
        self._devices: Dict[str, SDRDevice] = {}
        self._detected: List[DetectedDevice] = []

    def scan_devices(self) -> List[DetectedDevice]:
        """
        Scan for all available SDR devices.

        Returns:
            List of detected devices
        """
        self._detected.clear()

        # Scan RTL-SDR devices
        try:
            rtl_devices = RTLSDRDevice.list_devices()
            for info in rtl_devices:
                self._detected.append(
                    DetectedDevice(
                        info=info, device_class=RTLSDRDevice, is_available=True
                    )
                )
                logger.info(f"Found RTL-SDR: {info.serial}")
        except Exception as e:
            logger.warning(f"Error scanning RTL-SDR devices: {e}")

        # Scan HackRF devices
        try:
            hackrf_devices = HackRFDevice.list_devices()
            for info in hackrf_devices:
                self._detected.append(
                    DetectedDevice(
                        info=info, device_class=HackRFDevice, is_available=True
                    )
                )
                logger.info(f"Found HackRF: {info.serial}")
        except Exception as e:
            logger.warning(f"Error scanning HackRF devices: {e}")

        # Scan MX-K2 keyer devices
        try:
            keyer_devices = MXK2Keyer.list_devices()
            for info in keyer_devices:
                self._detected.append(
                    DetectedDevice(
                        info=info, device_class=MXK2Keyer, is_available=True
                    )
                )
                logger.info(f"Found MX-K2 Keyer: {info.serial}")
        except Exception as e:
            logger.warning(f"Error scanning MX-K2 keyer devices: {e}")

        logger.info(f"Total devices found: {len(self._detected)}")
        return self._detected

    @property
    def detected_devices(self) -> List[DetectedDevice]:
        """Get list of detected devices from last scan."""
        return self._detected.copy()

    def get_device(self, device_id: str) -> Optional[SDRDevice]:
        """
        Get an open device by ID.

        Args:
            device_id: Device identifier (e.g., "rtlsdr_0", "hackrf_0")

        Returns:
            SDRDevice instance or None if not found
        """
        with self._lock:
            return self._devices.get(device_id)

    def create_device(self, device_type: str) -> Optional[SDRDevice]:
        """
        Create a new device instance.

        Args:
            device_type: Type of device ("rtlsdr" or "hackrf")

        Returns:
            New SDRDevice instance or None if type unknown
        """
        device_class = self.DEVICE_TYPES.get(device_type.lower())
        if device_class is None:
            logger.error(f"Unknown device type: {device_type}")
            return None
        return device_class()

    def open_device(
        self, device_type: str, index: int = 0, config: Optional[DeviceConfig] = None
    ) -> Optional[SDRDevice]:
        """
        Open and configure an SDR device.

        Args:
            device_type: Type of device ("rtlsdr" or "hackrf")
            index: Device index
            config: Optional device configuration

        Returns:
            Configured SDRDevice instance or None on failure
        """
        device = self.create_device(device_type)
        if device is None:
            return None

        if not device.open(index):
            logger.error(f"Failed to open {device_type} device {index}")
            # Clean up device on open failure to prevent resource leak
            try:
                device.close()
            except Exception as e:
                logger.debug(f"Error during device cleanup after open failure: {e}")
            return None

        # Apply configuration if provided
        if config is not None:
            self.apply_config(device, config)

        # Register device
        device_id = f"{device_type}_{index}"
        with self._lock:
            self._devices[device_id] = device

        logger.info(f"Opened device: {device_id}")
        return device

    def close_device(self, device_id: str) -> bool:
        """
        Close a device by ID.

        Args:
            device_id: Device identifier

        Returns:
            True if closed successfully
        """
        with self._lock:
            device = self._devices.pop(device_id, None)
        if device is None:
            logger.warning(f"Device not found: {device_id}")
            return False

        device.close()
        logger.info(f"Closed device: {device_id}")
        return True

    def close_all(self) -> None:
        """Close all open devices."""
        with self._lock:
            device_ids = list(self._devices.keys())
        for device_id in device_ids:
            self.close_device(device_id)

    def apply_config(self, device: SDRDevice, config: DeviceConfig) -> bool:
        """
        Apply configuration to a device.

        Args:
            device: SDR device instance
            config: Configuration to apply

        Returns:
            True if all settings applied successfully
        """
        success = True

        if not device.set_frequency(config.frequency):
            logger.warning(f"Failed to set frequency to {config.frequency}")
            success = False

        if not device.set_sample_rate(config.sample_rate):
            logger.warning(f"Failed to set sample rate to {config.sample_rate}")
            success = False

        if not device.set_bandwidth(config.bandwidth):
            logger.warning(f"Failed to set bandwidth to {config.bandwidth}")
            success = False

        if config.gain_mode == "auto":
            device.set_gain_mode(True)
        else:
            device.set_gain_mode(False)
            if not device.set_gain(config.gain):
                logger.warning(f"Failed to set gain to {config.gain}")
                success = False

        # Optional features
        if config.bias_tee:
            try:
                device.set_bias_tee(True)
            except NotImplementedError:
                pass

        if config.amp_enabled:
            try:
                device.set_amp(True)
            except NotImplementedError:
                pass

        # HackRF specific
        if isinstance(device, HackRFDevice):
            if hasattr(config, "lna_gain"):
                device.set_lna_gain(int(config.lna_gain))
            if hasattr(config, "vga_gain"):
                device.set_vga_gain(int(config.vga_gain))
            if hasattr(config, "tx_vga_gain"):
                device.set_tx_gain(config.tx_vga_gain)

        return success

    def get_rtlsdr(self, index: int = 0) -> Optional[RTLSDRDevice]:
        """Convenience method to get/open RTL-SDR device."""
        device_id = f"rtlsdr_{index}"
        with self._lock:
            device = self._devices.get(device_id)
        if device is None:
            device = self.open_device("rtlsdr", index)
        return cast(Optional[RTLSDRDevice], device)

    def get_hackrf(self, index: int = 0) -> Optional[HackRFDevice]:
        """Convenience method to get/open HackRF device."""
        device_id = f"hackrf_{index}"
        with self._lock:
            device = self._devices.get(device_id)
        if device is None:
            device = self.open_device("hackrf", index)
        return cast(Optional[HackRFDevice], device)

    def has_rtlsdr(self) -> bool:
        """Check if an RTL-SDR device is available."""
        for d in self._detected:
            if d.device_class == RTLSDRDevice and d.is_available:
                return True
        return False

    def has_hackrf(self) -> bool:
        """Check if a HackRF device is available."""
        for d in self._detected:
            if d.device_class == HackRFDevice and d.is_available:
                return True
        return False

    def has_dual_sdr(self) -> bool:
        """Check if both RTL-SDR and HackRF are available."""
        return self.has_rtlsdr() and self.has_hackrf()

    def has_mxk2_keyer(self) -> bool:
        """Check if an MX-K2 keyer is available."""
        for d in self._detected:
            if d.device_class == MXK2Keyer and d.is_available:
                return True
        return False

    def get_mxk2_keyer(self, index: int = 0) -> Optional[MXK2Keyer]:
        """Convenience method to get/open MX-K2 keyer device."""
        device_id = f"mxk2_keyer_{index}"
        with self._lock:
            device = self._devices.get(device_id)
        if device is None:
            device = self.open_device("mxk2_keyer", index)
        return cast(Optional[MXK2Keyer], device)

    def apply_keyer_config(self, keyer: MXK2Keyer, config: KeyerConfig) -> bool:
        """
        Apply keyer-specific configuration.

        Args:
            keyer: MX-K2 keyer instance
            config: Keyer configuration to apply

        Returns:
            True if all settings applied successfully
        """
        success = True

        if not keyer.set_wpm(config.wpm):
            logger.warning(f"Failed to set keyer WPM to {config.wpm}")
            success = False

        if not keyer.set_sidetone(config.sidetone_freq, config.sidetone_enabled):
            logger.warning(f"Failed to set sidetone to {config.sidetone_freq}")
            success = False

        # Map paddle mode string to enum
        from ..devices.mxk2_keyer import PaddleMode

        paddle_mode_map = {
            "iambic_a": PaddleMode.IAMBIC_A,
            "iambic_b": PaddleMode.IAMBIC_B,
            "ultimatic": PaddleMode.ULTIMATIC,
            "bug": PaddleMode.BUG,
            "straight": PaddleMode.STRAIGHT,
        }
        paddle_mode = paddle_mode_map.get(config.paddle_mode.lower(), PaddleMode.IAMBIC_B)
        if not keyer.set_paddle_mode(paddle_mode):
            logger.warning(f"Failed to set paddle mode to {config.paddle_mode}")
            success = False

        if not keyer.set_weight(config.weight):
            logger.warning(f"Failed to set weight to {config.weight}")
            success = False

        if config.paddle_swap:
            keyer.swap_paddles()

        return success

    @property
    def open_devices(self) -> Dict[str, SDRDevice]:
        """Get dictionary of currently open devices."""
        with self._lock:
            return self._devices.copy()

    def __enter__(self):
        """Context manager entry."""
        self.scan_devices()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_all()
        return False

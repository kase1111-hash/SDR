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
from ..devices.rtlsdr import RTLSDRDevice
from .config import DeviceConfig

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

    Thread Safety:
        - The _devices dictionary is protected by _lock (RLock)
        - get_device(), get_rtlsdr(), get_hackrf() are thread-safe
        - open_device() and close_device() are thread-safe
        - scan_devices() modifies _detected list and should be called from main thread
        - apply_config() is not thread-safe; caller must ensure exclusive device access
        - The open_devices property returns a copy to prevent external modification
    """

    # Registered device types
    DEVICE_TYPES: Dict[str, Type[SDRDevice]] = {
        "rtlsdr": RTLSDRDevice,
        "hackrf": HackRFDevice,
    }

    def __init__(self) -> None:
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

    def _open_device_unlocked(
        self, device_type: str, index: int = 0, config: Optional[DeviceConfig] = None
    ) -> Optional[SDRDevice]:
        """
        Open and configure an SDR device (caller must hold _lock).

        Internal method used by get_rtlsdr/get_hackrf to prevent TOCTOU race.
        The caller must already hold self._lock before calling this method.

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

        # Register device (lock is already held by caller)
        device_id = f"{device_type}_{index}"
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

    def apply_config(
        self, device: SDRDevice, config: DeviceConfig, fail_fast: bool = False
    ) -> bool:
        """
        Apply configuration to a device.

        Args:
            device: SDR device instance
            config: Configuration to apply
            fail_fast: If True, stop on first failure; if False, try all settings

        Returns:
            True if all settings applied successfully
        """
        failures = []

        if not device.set_frequency(config.frequency):
            failures.append(f"frequency={config.frequency}")
            if fail_fast:
                logger.error(
                    f"Config failed: could not set frequency to {config.frequency}"
                )
                return False

        if not device.set_sample_rate(config.sample_rate):
            failures.append(f"sample_rate={config.sample_rate}")
            if fail_fast:
                logger.error(
                    f"Config failed: could not set sample rate to {config.sample_rate}"
                )
                return False

        if not device.set_bandwidth(config.bandwidth):
            failures.append(f"bandwidth={config.bandwidth}")
            if fail_fast:
                logger.error(
                    f"Config failed: could not set bandwidth to {config.bandwidth}"
                )
                return False

        if config.gain_mode == "auto":
            device.set_gain_mode(True)
        else:
            device.set_gain_mode(False)
            if not device.set_gain(config.gain):
                failures.append(f"gain={config.gain}")
                if fail_fast:
                    logger.error(f"Config failed: could not set gain to {config.gain}")
                    return False

        # Optional features
        if config.bias_tee:
            try:
                if not device.set_bias_tee(True):
                    failures.append("bias_tee=True")
            except NotImplementedError:
                pass  # Feature not supported, not a failure

        if config.amp_enabled:
            try:
                if not device.set_amp(True):
                    failures.append("amp=True")
            except NotImplementedError:
                pass  # Feature not supported, not a failure

        # HackRF specific
        if isinstance(device, HackRFDevice):
            if hasattr(config, "lna_gain"):
                if not device.set_lna_gain(int(config.lna_gain)):
                    failures.append(f"lna_gain={config.lna_gain}")
            if hasattr(config, "vga_gain"):
                if not device.set_vga_gain(int(config.vga_gain)):
                    failures.append(f"vga_gain={config.vga_gain}")
            if hasattr(config, "tx_vga_gain"):
                if not device.set_tx_gain(config.tx_vga_gain):
                    failures.append(f"tx_vga_gain={config.tx_vga_gain}")

        if failures:
            logger.warning(
                f"Config partially applied, failed settings: {', '.join(failures)}"
            )
            return False

        return True

    def get_rtlsdr(self, index: int = 0) -> Optional[RTLSDRDevice]:
        """Convenience method to get/open RTL-SDR device.

        Thread-safe: uses lock to prevent race conditions when multiple
        threads try to open the same device simultaneously.
        """
        device_id = f"rtlsdr_{index}"
        with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                # Open device while still holding lock to prevent TOCTOU race
                device = self._open_device_unlocked("rtlsdr", index)
        return cast(Optional[RTLSDRDevice], device)

    def get_hackrf(self, index: int = 0) -> Optional[HackRFDevice]:
        """Convenience method to get/open HackRF device.

        Thread-safe: uses lock to prevent race conditions when multiple
        threads try to open the same device simultaneously.
        """
        device_id = f"hackrf_{index}"
        with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                # Open device while still holding lock to prevent TOCTOU race
                device = self._open_device_unlocked("hackrf", index)
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

    @property
    def open_devices(self) -> Dict[str, SDRDevice]:
        """Get dictionary of currently open devices."""
        with self._lock:
            return self._devices.copy()

    def __enter__(self) -> "DeviceManager":
        """Context manager entry."""
        self.scan_devices()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> bool:
        """Context manager exit."""
        self.close_all()
        return False

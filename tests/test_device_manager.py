"""Tests for device manager."""

from unittest.mock import patch

from sdr_module.core.config import DeviceConfig
from sdr_module.core.device_manager import DeviceManager
from sdr_module.devices.base import DeviceCapability, DeviceInfo, SDRDevice


class MockSDRDevice(SDRDevice):
    """Mock SDR device for testing."""

    def __init__(self):
        super().__init__()
        self._info = DeviceInfo(
            name="MockSDR",
            serial="12345",
            manufacturer="Test",
            product="MockSDR",
            index=0,
            capabilities=[DeviceCapability.RX],
        )
        self._is_open = False

    def open(self, index: int = 0) -> bool:
        self._is_open = True
        return True

    def close(self) -> None:
        self._is_open = False

    def set_frequency(self, freq_hz: float) -> bool:
        self._state.frequency = freq_hz
        return True

    def set_sample_rate(self, rate_hz: float) -> bool:
        self._state.sample_rate = rate_hz
        return True

    def set_bandwidth(self, bw_hz: float) -> bool:
        self._state.bandwidth = bw_hz
        return True

    def set_gain(self, gain_db: float) -> bool:
        self._state.gain = gain_db
        return True

    def set_gain_mode(self, auto: bool) -> bool:
        self._state.gain_mode = "auto" if auto else "manual"
        return True

    def start_rx(self, callback=None) -> bool:
        self._state.is_streaming = True
        self._rx_callback = callback
        return True

    def stop_rx(self) -> bool:
        self._state.is_streaming = False
        return True


class TestDeviceManager:
    """Test device manager functionality."""

    def test_initialization(self):
        """Test device manager initialization."""
        manager = DeviceManager()
        assert manager is not None
        assert len(manager.detected_devices) == 0
        assert len(manager.open_devices) == 0

    @patch("sdr_module.devices.rtlsdr.RTLSDRDevice.list_devices")
    @patch("sdr_module.devices.hackrf.HackRFDevice.list_devices")
    def test_scan_devices_none_found(self, mock_hackrf_list, mock_rtl_list):
        """Test scanning when no devices are found."""
        mock_rtl_list.return_value = []
        mock_hackrf_list.return_value = []

        manager = DeviceManager()
        detected = manager.scan_devices()

        assert len(detected) == 0
        assert not manager.has_rtlsdr()
        assert not manager.has_hackrf()
        assert not manager.has_dual_sdr()

    @patch("sdr_module.devices.rtlsdr.RTLSDRDevice.list_devices")
    @patch("sdr_module.devices.hackrf.HackRFDevice.list_devices")
    def test_scan_devices_rtlsdr_only(self, mock_hackrf_list, mock_rtl_list):
        """Test scanning with only RTL-SDR found."""
        rtl_info = DeviceInfo(
            name="RTL-SDR",
            serial="00000001",
            manufacturer="Realtek",
            product="RTL2838UHIDIR",
            index=0,
        )
        mock_rtl_list.return_value = [rtl_info]
        mock_hackrf_list.return_value = []

        manager = DeviceManager()
        detected = manager.scan_devices()

        assert len(detected) == 1
        assert manager.has_rtlsdr()
        assert not manager.has_hackrf()
        assert not manager.has_dual_sdr()

    @patch("sdr_module.devices.rtlsdr.RTLSDRDevice.list_devices")
    @patch("sdr_module.devices.hackrf.HackRFDevice.list_devices")
    def test_scan_devices_both_found(self, mock_hackrf_list, mock_rtl_list):
        """Test scanning with both devices found."""
        rtl_info = DeviceInfo(
            name="RTL-SDR",
            serial="00000001",
            manufacturer="Realtek",
            product="RTL2838UHIDIR",
            index=0,
        )
        hackrf_info = DeviceInfo(
            name="HackRF One",
            serial="0000000000000000a06063c8234e5f8f",
            manufacturer="Great Scott Gadgets",
            product="HackRF One",
            index=0,
        )

        mock_rtl_list.return_value = [rtl_info]
        mock_hackrf_list.return_value = [hackrf_info]

        manager = DeviceManager()
        detected = manager.scan_devices()

        assert len(detected) == 2
        assert manager.has_rtlsdr()
        assert manager.has_hackrf()
        assert manager.has_dual_sdr()

    def test_create_device(self):
        """Test creating device instances."""
        manager = DeviceManager()

        # Should create device objects (though they may fail to open without hardware)
        rtlsdr_device = manager.create_device("rtlsdr")
        assert rtlsdr_device is not None

        hackrf_device = manager.create_device("hackrf")
        assert hackrf_device is not None

    def test_create_unknown_device(self):
        """Test creating unknown device type."""
        manager = DeviceManager()
        device = manager.create_device("unknown_device")
        assert device is None

    def test_close_all(self):
        """Test closing all devices."""
        manager = DeviceManager()

        # Add mock devices
        mock_dev1 = MockSDRDevice()
        mock_dev2 = MockSDRDevice()

        manager._devices["dev1"] = mock_dev1
        manager._devices["dev2"] = mock_dev2

        mock_dev1.open()
        mock_dev2.open()

        manager.close_all()

        assert not mock_dev1.is_open
        assert not mock_dev2.is_open
        assert len(manager.open_devices) == 0

    def test_apply_config(self):
        """Test applying configuration to device."""
        manager = DeviceManager()
        device = MockSDRDevice()
        device.open()

        config = DeviceConfig(
            frequency=144.5e6,
            sample_rate=2.4e6,
            bandwidth=2e6,
            gain=30.0,
            gain_mode="manual",
        )

        success = manager.apply_config(device, config)
        assert success
        assert device.state.frequency == 144.5e6
        assert device.state.sample_rate == 2.4e6
        assert device.state.bandwidth == 2e6
        assert device.state.gain == 30.0
        assert device.state.gain_mode == "manual"

    def test_apply_config_auto_gain(self):
        """Test applying auto gain configuration."""
        manager = DeviceManager()
        device = MockSDRDevice()
        device.open()

        config = DeviceConfig(
            frequency=144.5e6,
            sample_rate=2.4e6,
            bandwidth=2e6,
            gain=0,
            gain_mode="auto",
        )

        success = manager.apply_config(device, config)
        assert success
        assert device.state.gain_mode == "auto"

    def test_get_device(self):
        """Test getting device by ID."""
        manager = DeviceManager()
        mock_dev = MockSDRDevice()
        manager._devices["test_dev"] = mock_dev

        retrieved = manager.get_device("test_dev")
        assert retrieved is mock_dev

        not_found = manager.get_device("nonexistent")
        assert not_found is None

    def test_close_device(self):
        """Test closing specific device."""
        manager = DeviceManager()
        mock_dev = MockSDRDevice()
        mock_dev.open()
        manager._devices["test_dev"] = mock_dev

        success = manager.close_device("test_dev")
        assert success
        assert not mock_dev.is_open
        assert "test_dev" not in manager._devices

    def test_close_nonexistent_device(self):
        """Test closing non-existent device."""
        manager = DeviceManager()
        success = manager.close_device("nonexistent")
        assert not success

    def test_context_manager(self):
        """Test using device manager as context manager."""
        with (
            patch("sdr_module.devices.rtlsdr.RTLSDRDevice.list_devices"),
            patch("sdr_module.devices.hackrf.HackRFDevice.list_devices"),
        ):

            with DeviceManager() as manager:
                assert manager is not None


class TestDeviceManagerIntegration:
    """Integration tests for device manager."""

    def test_device_lifecycle(self):
        """Test complete device lifecycle."""
        manager = DeviceManager()

        # Create mock device
        mock_dev = MockSDRDevice()
        manager._devices["mock_0"] = mock_dev

        # Open
        mock_dev.open()
        assert mock_dev.is_open

        # Configure
        config = DeviceConfig(
            frequency=100e6,
            sample_rate=2e6,
            bandwidth=2e6,
            gain=20,
        )
        manager.apply_config(mock_dev, config)

        # Verify configuration
        assert mock_dev.state.frequency == 100e6

        # Close
        manager.close_device("mock_0")
        assert not mock_dev.is_open

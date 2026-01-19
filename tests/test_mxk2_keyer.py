"""Tests for MX-K2 CW Keyer device driver."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from sdr_module.core.config import KeyerConfig
from sdr_module.core.device_manager import DeviceManager
from sdr_module.devices.base import DeviceCapability
from sdr_module.devices.mxk2_keyer import (
    KeyerState,
    KeyerStatus,
    MXK2Command,
    MXK2Config,
    MXK2Keyer,
    PaddleMode,
)

# Create a mock serial module for tests
mock_serial_module = MagicMock()
sys.modules["serial"] = mock_serial_module
sys.modules["serial.tools"] = MagicMock()
sys.modules["serial.tools.list_ports"] = MagicMock()


class TestMXK2Config:
    """Test MX-K2 configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MXK2Config()
        assert config.port == ""
        assert config.baud_rate == 1200
        assert config.wpm == 20
        assert config.sidetone_freq == 700
        assert config.sidetone_enabled is True
        assert config.paddle_mode == PaddleMode.IAMBIC_B
        assert config.paddle_swap is False
        assert config.weight == 50
        assert config.dah_to_dit_ratio == 3.0
        assert config.ptt_lead_time_ms == 50
        assert config.ptt_tail_time_ms == 100
        assert config.auto_space is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MXK2Config(
            port="/dev/ttyUSB0",
            wpm=25,
            sidetone_freq=800,
            paddle_mode=PaddleMode.IAMBIC_A,
            weight=55,
        )
        assert config.port == "/dev/ttyUSB0"
        assert config.wpm == 25
        assert config.sidetone_freq == 800
        assert config.paddle_mode == PaddleMode.IAMBIC_A
        assert config.weight == 55


class TestKeyerStatus:
    """Test keyer status dataclass."""

    def test_default_status(self):
        """Test default status values."""
        status = KeyerStatus()
        assert status.is_connected is False
        assert status.is_keying is False
        assert status.wpm == 20
        assert status.sidetone_freq == 700
        assert status.paddle_mode == PaddleMode.IAMBIC_B
        assert status.ptt_active is False
        assert status.key_down is False
        assert status.state == KeyerState.IDLE


class TestPaddleMode:
    """Test paddle mode enumeration."""

    def test_paddle_modes(self):
        """Test all paddle modes exist."""
        assert PaddleMode.IAMBIC_A is not None
        assert PaddleMode.IAMBIC_B is not None
        assert PaddleMode.ULTIMATIC is not None
        assert PaddleMode.BUG is not None
        assert PaddleMode.STRAIGHT is not None


class TestMXK2Keyer:
    """Test MX-K2 Keyer device driver."""

    def test_initialization(self):
        """Test keyer initialization."""
        keyer = MXK2Keyer()
        assert keyer is not None
        assert keyer.is_open is False
        assert keyer._config is not None
        assert keyer._status is not None

    def test_initialization_with_config(self):
        """Test keyer initialization with custom config."""
        config = MXK2Config(wpm=30, sidetone_freq=600)
        keyer = MXK2Keyer(config)
        assert keyer._config.wpm == 30
        assert keyer._config.sidetone_freq == 600

    def test_device_capabilities(self):
        """Test device reports correct capabilities."""
        keyer = MXK2Keyer()
        # Create mock info for testing
        keyer._info = MagicMock()
        keyer._info.capabilities = [
            DeviceCapability.TX,
            DeviceCapability.HALF_DUPLEX,
        ]
        assert keyer.has_capability(DeviceCapability.TX)
        assert keyer.has_capability(DeviceCapability.HALF_DUPLEX)
        assert not keyer.has_capability(DeviceCapability.RX)

    def test_open_close(self):
        """Test opening and closing the keyer."""
        mock_serial = MagicMock()
        mock_serial.read_until.return_value = b"MX-K2 v1.0\r"
        mock_serial_module.Serial.return_value = mock_serial

        keyer = MXK2Keyer(MXK2Config(port="/dev/ttyUSB0"))

        # Open
        result = keyer.open(port="/dev/ttyUSB0")
        assert result is True
        assert keyer.is_open is True

        # Close
        keyer.close()
        assert keyer.is_open is False
        mock_serial.close.assert_called_once()

    def test_sdr_interface_methods_no_op(self):
        """Test that SDR interface methods are no-ops for keyer."""
        keyer = MXK2Keyer()
        # These methods should return True but do nothing
        assert keyer.set_frequency(100e6) is True
        assert keyer.set_sample_rate(2.4e6) is True
        assert keyer.set_bandwidth(200e3) is True
        assert keyer.set_gain(30) is True
        assert keyer.set_gain_mode(True) is True

    def test_start_rx_not_supported(self):
        """Test that RX is not supported."""
        keyer = MXK2Keyer()
        assert keyer.start_rx() is False

    def test_set_wpm(self):
        """Test setting keyer speed."""
        mock_serial = MagicMock()
        mock_serial.read_until.return_value = b"\r"

        keyer = MXK2Keyer(MXK2Config(port="/dev/ttyUSB0"))
        keyer._serial = mock_serial
        keyer._is_open = True

        result = keyer.set_wpm(25)
        assert result is True
        assert keyer._config.wpm == 25
        assert keyer._status.wpm == 25

    def test_set_wpm_clamp_min(self):
        """Test WPM is clamped to minimum value."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        keyer.set_wpm(3)  # Below minimum
        assert keyer._config.wpm == 5  # Clamped to minimum

    def test_set_wpm_clamp_max(self):
        """Test WPM is clamped to maximum value."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        keyer.set_wpm(60)  # Above maximum
        assert keyer._config.wpm == 50  # Clamped to maximum

    def test_set_sidetone(self):
        """Test setting sidetone configuration."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        result = keyer.set_sidetone(800, True)
        assert result is True
        assert keyer._config.sidetone_freq == 800
        assert keyer._config.sidetone_enabled is True

    def test_set_paddle_mode(self):
        """Test setting paddle mode."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        result = keyer.set_paddle_mode(PaddleMode.IAMBIC_A)
        assert result is True
        assert keyer._config.paddle_mode == PaddleMode.IAMBIC_A

    def test_send_text(self):
        """Test sending CW text."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        result = keyer.send_text("CQ CQ DE W1AW")
        assert result is True
        assert keyer._status.is_keying is True

    def test_send_text_filters_invalid_chars(self):
        """Test that invalid characters are filtered from text."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        # Text with invalid characters should still be sent (filtered)
        result = keyer.send_text("CQ @#$ DE W1AW!")
        assert result is True

    def test_abort(self):
        """Test aborting transmission."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True
        keyer._status.is_keying = True

        result = keyer.abort()
        assert result is True
        assert keyer._status.is_keying is False
        assert keyer._status.state == KeyerState.IDLE

    def test_key_down_up(self):
        """Test manual key control."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        # Key down
        result = keyer.key_down()
        assert result is True
        assert keyer._status.key_down is True

        # Key up
        result = keyer.key_up()
        assert result is True
        assert keyer._status.key_down is False

    def test_ptt_control(self):
        """Test PTT control."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        # PTT on
        result = keyer.ptt_on()
        assert result is True
        assert keyer._status.ptt_active is True

        # PTT off
        result = keyer.ptt_off()
        assert result is True
        assert keyer._status.ptt_active is False

    def test_memory_operations(self):
        """Test memory slot operations."""
        mock_serial = MagicMock()

        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        # Store memory
        result = keyer.store_memory(1, "CQ TEST")
        assert result is True
        assert keyer._config.memory_slots[0] == "CQ TEST"

        # Play memory
        result = keyer.play_memory(1)
        assert result is True
        assert keyer._status.memory_slot == 1
        assert keyer._status.state == KeyerState.MEMORY_PLAYBACK

    def test_memory_slot_validation(self):
        """Test memory slot validation."""
        keyer = MXK2Keyer()
        keyer._is_open = True

        # Invalid slots
        assert keyer.play_memory(0) is False
        assert keyer.play_memory(5) is False
        assert keyer.store_memory(0, "TEST") is False
        assert keyer.store_memory(5, "TEST") is False

    def test_callbacks(self):
        """Test callback registration."""
        keyer = MXK2Keyer()

        key_callback = MagicMock()
        ptt_callback = MagicMock()

        keyer.set_on_key_change(key_callback)
        keyer.set_on_ptt_change(ptt_callback)

        assert keyer._on_key_change == key_callback
        assert keyer._on_ptt_change == ptt_callback

    def test_get_status(self):
        """Test getting keyer status."""
        keyer = MXK2Keyer()
        status = keyer.get_status()
        assert isinstance(status, KeyerStatus)

    def test_get_config(self):
        """Test getting keyer configuration."""
        keyer = MXK2Keyer()
        config = keyer.get_config()
        assert isinstance(config, MXK2Config)

    def test_repr_not_connected(self):
        """Test string representation when not connected."""
        keyer = MXK2Keyer()
        assert "not connected" in repr(keyer)

    def test_repr_connected(self):
        """Test string representation when connected."""
        keyer = MXK2Keyer()
        keyer._info = MagicMock()
        keyer._info.serial = "/dev/ttyUSB0"
        keyer._status.wpm = 25
        assert "MXK2Keyer" in repr(keyer)
        assert "25 WPM" in repr(keyer)


class TestMXK2KeyerDeviceManager:
    """Test MX-K2 integration with device manager."""

    def test_device_type_registered(self):
        """Test MX-K2 is registered in device types."""
        manager = DeviceManager()
        assert "mxk2_keyer" in manager.DEVICE_TYPES
        assert manager.DEVICE_TYPES["mxk2_keyer"] == MXK2Keyer

    def test_create_mxk2_device(self):
        """Test creating MX-K2 device through manager."""
        manager = DeviceManager()
        device = manager.create_device("mxk2_keyer")
        assert device is not None
        assert isinstance(device, MXK2Keyer)

    @patch("sdr_module.devices.mxk2_keyer.MXK2Keyer.list_devices")
    @patch("sdr_module.devices.rtlsdr.RTLSDRDevice.list_devices")
    @patch("sdr_module.devices.hackrf.HackRFDevice.list_devices")
    def test_scan_finds_keyer(
        self, mock_hackrf_list, mock_rtl_list, mock_keyer_list
    ):
        """Test that scanning finds MX-K2 keyer."""
        from sdr_module.devices.base import DeviceInfo

        mock_rtl_list.return_value = []
        mock_hackrf_list.return_value = []
        mock_keyer_list.return_value = [
            DeviceInfo(
                name="MX-K2 CW Keyer",
                serial="/dev/ttyUSB0",
                manufacturer="MFJ Enterprises",
                product="MX-K2",
                index=0,
                capabilities=[DeviceCapability.TX, DeviceCapability.HALF_DUPLEX],
            )
        ]

        manager = DeviceManager()
        detected = manager.scan_devices()

        assert len(detected) == 1
        assert manager.has_mxk2_keyer()

    def test_has_mxk2_keyer_false(self):
        """Test has_mxk2_keyer returns False when no keyer."""
        manager = DeviceManager()
        manager._detected = []
        assert manager.has_mxk2_keyer() is False


class TestKeyerConfigIntegration:
    """Test KeyerConfig integration."""

    def test_keyer_config_defaults(self):
        """Test KeyerConfig default values."""
        config = KeyerConfig()
        assert config.device_type == "mxk2_keyer"
        assert config.wpm == 20
        assert config.sidetone_freq == 700
        assert config.paddle_mode == "iambic_b"

    def test_apply_keyer_config(self):
        """Test applying KeyerConfig through device manager."""
        mock_serial = MagicMock()

        manager = DeviceManager()
        keyer = MXK2Keyer()
        keyer._serial = mock_serial
        keyer._is_open = True

        config = KeyerConfig(
            wpm=30,
            sidetone_freq=600,
            paddle_mode="iambic_a",
            weight=45,
        )

        result = manager.apply_keyer_config(keyer, config)
        assert result is True
        assert keyer._config.wpm == 30
        assert keyer._config.sidetone_freq == 600
        assert keyer._config.paddle_mode == PaddleMode.IAMBIC_A
        assert keyer._config.weight == 45


class TestMXK2Commands:
    """Test MX-K2 command constants."""

    def test_command_bytes(self):
        """Test command constants are bytes."""
        assert isinstance(MXK2Command.SET_SPEED, bytes)
        assert isinstance(MXK2Command.SET_SIDETONE, bytes)
        assert isinstance(MXK2Command.SEND_TEXT, bytes)
        assert isinstance(MXK2Command.KEY_DOWN, bytes)
        assert isinstance(MXK2Command.KEY_UP, bytes)
        assert isinstance(MXK2Command.PTT_ON, bytes)
        assert isinstance(MXK2Command.PTT_OFF, bytes)
        assert isinstance(MXK2Command.ABORT, bytes)

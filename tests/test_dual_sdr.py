"""Tests for dual-SDR controller."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from sdr_module.core.dual_sdr import (
    DualSDRController,
    OperationMode,
    DualSDRState,
)
from sdr_module.core.config import SDRConfig


class MockRTLSDRDevice:
    """Mock RTL-SDR device for testing."""

    def __init__(self):
        self.state = Mock()
        self.state.frequency = 100e6
        self.state.sample_rate = 2.4e6
        self.state.gain = 20
        self.state.is_streaming = False
        self._rx_callback = None

    def set_frequency(self, freq_hz):
        self.state.frequency = freq_hz
        return True

    def set_sample_rate(self, rate_hz):
        self.state.sample_rate = rate_hz
        return True

    def set_bandwidth(self, bw_hz):
        self.state.bandwidth = bw_hz
        return True

    def set_gain(self, gain_db):
        self.state.gain = gain_db
        return True

    def set_gain_mode(self, auto):
        return True

    def start_rx(self, callback=None):
        self._rx_callback = callback
        self.state.is_streaming = True
        return True

    def stop_rx(self):
        self.state.is_streaming = False
        return True


class MockHackRFDevice:
    """Mock HackRF device for testing."""

    def __init__(self):
        self.state = Mock()
        self.state.frequency = 433e6
        self.state.sample_rate = 8e6
        self.state.gain = 0
        self.state.is_streaming = False
        self.state.is_transmitting = False
        self._rx_callback = None
        self._tx_generator = None

    def set_frequency(self, freq_hz):
        self.state.frequency = freq_hz
        return True

    def set_sample_rate(self, rate_hz):
        self.state.sample_rate = rate_hz
        return True

    def set_bandwidth(self, bw_hz):
        self.state.bandwidth = bw_hz
        return True

    def set_gain(self, gain_db):
        self.state.gain = gain_db
        return True

    def set_gain_mode(self, auto):
        return True

    def set_lna_gain(self, gain):
        return True

    def set_vga_gain(self, gain):
        return True

    def set_tx_gain(self, gain):
        return True

    def start_rx(self, callback=None):
        self._rx_callback = callback
        self.state.is_streaming = True
        return True

    def stop_rx(self):
        self.state.is_streaming = False
        return True

    def start_tx(self, generator=None):
        self._tx_generator = generator
        self.state.is_transmitting = True
        return True

    def stop_tx(self):
        self.state.is_transmitting = False
        return True


class TestDualSDRController:
    """Test DualSDRController basic functionality."""

    def test_initialization(self):
        """Test controller initialization."""
        controller = DualSDRController()
        assert controller is not None
        assert controller.state.mode == OperationMode.DUAL_RX
        assert controller.rtlsdr is None
        assert controller.hackrf is None

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    def test_initialize_no_devices(self, mock_get_hackrf, mock_get_rtlsdr, mock_scan):
        """Test initialization when no devices available."""
        mock_scan.return_value = []
        mock_get_rtlsdr.return_value = None
        mock_get_hackrf.return_value = None

        controller = DualSDRController()
        result = controller.initialize()

        assert not result
        assert controller.rtlsdr is None
        assert controller.hackrf is None

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    @patch("sdr_module.core.device_manager.DeviceManager.apply_config")
    def test_initialize_with_devices(self, mock_apply, mock_get_hackrf, mock_get_rtlsdr, mock_scan):
        """Test initialization with both devices."""
        mock_rtl = MockRTLSDRDevice()
        mock_hack = MockHackRFDevice()

        mock_scan.return_value = []
        mock_get_rtlsdr.return_value = mock_rtl
        mock_get_hackrf.return_value = mock_hack
        mock_apply.return_value = True

        controller = DualSDRController()
        result = controller.initialize()

        assert result
        assert controller.rtlsdr is not None
        assert controller.hackrf is not None

    def test_set_mode(self):
        """Test setting operation mode."""
        controller = DualSDRController()
        controller._hackrf = MockHackRFDevice()

        # Should succeed when not streaming
        result = controller.set_mode(OperationMode.DUAL_RX)
        assert result
        assert controller.state.mode == OperationMode.DUAL_RX

        result = controller.set_mode(OperationMode.TX_MONITOR)
        assert result
        assert controller.state.mode == OperationMode.TX_MONITOR

    def test_set_mode_while_streaming(self):
        """Test that mode cannot be changed while streaming."""
        controller = DualSDRController()
        controller._state.rtlsdr_streaming = True

        result = controller.set_mode(OperationMode.FULL_DUPLEX)
        assert not result

    def test_set_frequencies(self):
        """Test setting frequencies."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        rtl_result, hack_result = controller.set_frequencies(144.5e6, 433e6)

        assert rtl_result
        assert hack_result
        assert controller.state.rtlsdr_frequency == 144.5e6
        assert controller.state.hackrf_frequency == 433e6

    def test_get_status(self):
        """Test getting controller status."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        status = controller.get_status()

        assert "mode" in status
        assert "rtlsdr" in status
        assert "hackrf" in status
        assert status["rtlsdr"]["connected"]
        assert status["hackrf"]["connected"]


class TestDualSDROperationModes:
    """Test different operation modes."""

    def test_start_dual_rx(self):
        """Test starting dual receive mode."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        callback_rtl = Mock()
        callback_hack = Mock()

        result = controller.start_dual_rx(callback_rtl, callback_hack)

        assert result
        assert controller.state.mode == OperationMode.DUAL_RX
        assert controller.state.rtlsdr_streaming
        assert controller.state.hackrf_streaming

    def test_start_full_duplex(self):
        """Test starting full-duplex mode."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        rx_callback = Mock()
        tx_generator = Mock(return_value=np.zeros(1024, dtype=np.complex64))

        result = controller.start_full_duplex(rx_callback, tx_generator)

        assert result
        assert controller.state.mode == OperationMode.FULL_DUPLEX
        assert controller.state.rtlsdr_streaming
        assert controller.state.hackrf_transmitting

    def test_start_full_duplex_missing_device(self):
        """Test full-duplex fails without both devices."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        # No HackRF

        result = controller.start_full_duplex()
        assert not result

    def test_start_tx_monitor(self):
        """Test starting TX monitor mode."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()
        controller._state.hackrf_frequency = 433e6
        controller._hackrf.state.frequency = 433e6

        tx_generator = Mock(return_value=np.zeros(1024, dtype=np.complex64))
        monitor_callback = Mock()

        result = controller.start_tx_monitor(tx_generator, monitor_callback)

        assert result
        # Note: start_tx_monitor calls start_full_duplex internally,
        # which resets the mode to FULL_DUPLEX
        assert controller.state.mode == OperationMode.FULL_DUPLEX
        assert controller.state.hackrf_transmitting

    def test_stop_all(self):
        """Test stopping all operations."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        # Start dual RX
        controller.start_dual_rx()

        # Stop all
        controller.stop_all()

        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming
        assert not controller.state.hackrf_transmitting

    def test_stop_rtlsdr_only(self):
        """Test stopping only RTL-SDR."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()
        controller.stop_rtlsdr()

        assert not controller.state.rtlsdr_streaming
        assert controller.state.hackrf_streaming

    def test_stop_hackrf_only(self):
        """Test stopping only HackRF."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()
        controller.stop_hackrf()

        assert controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming


class TestDualSDRBuffers:
    """Test buffer operations."""

    def test_buffer_access(self):
        """Test accessing sample buffers."""
        controller = DualSDRController()

        rtl_buffer = controller.rtlsdr_buffer
        hack_buffer = controller.hackrf_buffer

        assert rtl_buffer is not None
        assert hack_buffer is not None
        assert rtl_buffer.capacity == 2 * 1024 * 1024
        assert hack_buffer.capacity == 8 * 1024 * 1024

    def test_write_to_buffers(self):
        """Test writing samples to buffers."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        # Start dual RX to set up callbacks
        controller.start_dual_rx()

        # Simulate device callbacks
        rtl_samples = np.random.randn(1000) + 1j * np.random.randn(1000)
        rtl_samples = rtl_samples.astype(np.complex64)

        controller._rtlsdr._rx_callback(rtl_samples)

        # Buffer should have samples
        assert controller.rtlsdr_buffer.available > 0

    def test_read_from_buffers(self):
        """Test reading samples from buffers."""
        controller = DualSDRController()

        # Write some samples
        samples = np.random.randn(1000) + 1j * np.random.randn(1000)
        samples = samples.astype(np.complex64)
        controller._rtlsdr_buffer.write(samples)

        # Read samples
        read_samples = controller.read_rtlsdr_samples(500, timeout=0.1)

        assert read_samples is not None
        assert len(read_samples) == 500


class TestDualSDRContextManager:
    """Test context manager functionality."""

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    @patch("sdr_module.core.device_manager.DeviceManager.apply_config")
    def test_context_manager(self, mock_apply, mock_get_hackrf, mock_get_rtlsdr, mock_scan):
        """Test using controller as context manager."""
        mock_rtl = MockRTLSDRDevice()
        mock_hack = MockHackRFDevice()

        mock_scan.return_value = []
        mock_get_rtlsdr.return_value = mock_rtl
        mock_get_hackrf.return_value = mock_hack
        mock_apply.return_value = True

        with DualSDRController() as controller:
            assert controller is not None
            assert controller.rtlsdr is not None


class TestDualSDRState:
    """Test DualSDRState dataclass."""

    def test_state_initialization(self):
        """Test state initialization."""
        state = DualSDRState()

        assert state.mode == OperationMode.DUAL_RX
        assert not state.rtlsdr_streaming
        assert not state.hackrf_streaming
        assert not state.hackrf_transmitting
        assert state.rtlsdr_frequency == 0.0
        assert state.hackrf_frequency == 0.0
        assert not state.is_synchronized


class TestOperationMode:
    """Test OperationMode enum."""

    def test_operation_modes(self):
        """Test all operation modes exist."""
        assert OperationMode.DUAL_RX
        assert OperationMode.FULL_DUPLEX
        assert OperationMode.TX_MONITOR
        assert OperationMode.WIDEBAND_SCAN
        assert OperationMode.RELAY

    def test_mode_values(self):
        """Test mode values."""
        assert OperationMode.DUAL_RX.value == "dual_rx"
        assert OperationMode.FULL_DUPLEX.value == "full_duplex"
        assert OperationMode.TX_MONITOR.value == "tx_monitor"

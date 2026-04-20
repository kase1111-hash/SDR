"""Tests for DualSDRController failure modes and edge cases.

Covers device disconnection, buffer overrun, mode switching during
active streaming, initialization with no devices, double start/stop,
and frequency setting on uninitialized controllers.
"""

import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sdr_module.core.dual_sdr import (
    DualSDRController,
    OperationMode,
)
from sdr_module.core.sample_buffer import SampleBuffer

# ---------------------------------------------------------------------------
# Mock devices (reused from the main test suite with failure-injection hooks)
# ---------------------------------------------------------------------------


class MockRTLSDRDevice:
    """Mock RTL-SDR device with failure-injection support."""

    def __init__(self):
        self.state = Mock()
        self.state.frequency = 100e6
        self.state.sample_rate = 2.4e6
        self.state.gain = 20
        self.state.is_streaming = False
        self._rx_callback = None
        # Failure-injection flags
        self._fail_on_start = False
        self._fail_on_stop = False
        self._raise_on_callback = False
        self._callback_exception = RuntimeError("Device disconnected")

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
        if self._fail_on_start:
            return False
        self._rx_callback = callback
        self.state.is_streaming = True
        return True

    def stop_rx(self):
        if self._fail_on_stop:
            raise RuntimeError("Device disconnected during stop")
        self.state.is_streaming = False
        return True


class MockHackRFDevice:
    """Mock HackRF device with failure-injection support."""

    def __init__(self):
        self.state = Mock()
        self.state.frequency = 433e6
        self.state.sample_rate = 8e6
        self.state.gain = 0
        self.state.is_streaming = False
        self.state.is_transmitting = False
        self._rx_callback = None
        self._tx_generator = None
        # Failure-injection flags
        self._fail_on_start = False
        self._fail_on_stop = False
        self._raise_on_callback = False
        self._callback_exception = RuntimeError("Device disconnected")

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
        if self._fail_on_start:
            return False
        self._rx_callback = callback
        self.state.is_streaming = True
        return True

    def stop_rx(self):
        if self._fail_on_stop:
            raise RuntimeError("Device disconnected during stop")
        self.state.is_streaming = False
        return True

    def start_tx(self, generator=None):
        self._tx_generator = generator
        self.state.is_transmitting = True
        return True

    def stop_tx(self):
        self.state.is_transmitting = False
        return True


# ---------------------------------------------------------------------------
# Helper to generate random IQ samples
# ---------------------------------------------------------------------------


def _make_samples(n: int) -> np.ndarray:
    """Return *n* random complex64 IQ samples."""
    return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)


# ===========================================================================
# 1. Device disconnection during streaming
# ===========================================================================


class TestDeviceDisconnectionDuringStreaming:
    """Verify the controller recovers when a device raises mid-stream."""

    def test_rtlsdr_callback_raises_does_not_crash(self):
        """Simulate RTL-SDR raising in its RX callback.

        The internal callback that DualSDRController wraps around the
        user callback writes to the buffer first and then calls the
        user callback.  If the user callback raises, the controller
        must not hang or leave inconsistent state.
        """
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        exploding_callback = Mock(side_effect=RuntimeError("USB disconnect"))

        result = controller.start_dual_rx(
            rtlsdr_callback=exploding_callback,
            hackrf_callback=Mock(),
        )
        assert result

        # Feed samples into the RTL-SDR callback.  The internal wrapper
        # writes to the buffer and then invokes our exploding callback.
        samples = _make_samples(256)
        with pytest.raises(RuntimeError, match="USB disconnect"):
            controller._rtlsdr._rx_callback(samples)

        # Despite the exception, samples should have been written to the
        # buffer *before* the user callback was invoked.
        assert controller.rtlsdr_buffer.available == 256

        # Controller should still allow a clean stop.
        controller.stop_all()
        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming

    def test_hackrf_callback_raises_does_not_crash(self):
        """Same scenario but on the HackRF path."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        exploding_callback = Mock(side_effect=OSError("HackRF detached"))

        result = controller.start_dual_rx(
            rtlsdr_callback=Mock(),
            hackrf_callback=exploding_callback,
        )
        assert result

        samples = _make_samples(512)
        with pytest.raises(OSError, match="HackRF detached"):
            controller._hackrf._rx_callback(samples)

        assert controller.hackrf_buffer.available == 512

        controller.stop_all()
        assert not controller.state.hackrf_streaming
        assert not controller.state.rtlsdr_streaming

    def test_stop_all_after_disconnect_does_not_hang(self):
        """Verify stop_all completes within a reasonable time even if the
        device raises during stop_rx.
        """
        controller = DualSDRController()
        rtl = MockRTLSDRDevice()
        hack = MockHackRFDevice()
        controller._rtlsdr = rtl
        controller._hackrf = hack

        controller.start_dual_rx()
        assert controller.state.rtlsdr_streaming

        # Simulate device vanishing -- stop_rx now throws.
        rtl._fail_on_stop = True

        # stop_all should not hang.  We run it in a thread with a timeout.
        finished = threading.Event()

        def _do_stop():
            try:
                controller.stop_all()
            except RuntimeError:
                pass  # We expect the device error to propagate
            finally:
                finished.set()

        t = threading.Thread(target=_do_stop)
        t.start()
        assert finished.wait(timeout=5.0), "stop_all hung after device disconnect"
        t.join(timeout=1.0)

    def test_state_consistent_after_partial_start_failure(self):
        """If the RTL-SDR starts but the HackRF fails to start, state
        should reflect only what actually started.
        """
        controller = DualSDRController()
        rtl = MockRTLSDRDevice()
        hack = MockHackRFDevice()
        hack._fail_on_start = True

        controller._rtlsdr = rtl
        controller._hackrf = hack

        result = controller.start_dual_rx()

        # start_dual_rx returns False because HackRF failed.
        assert not result

        # RTL-SDR *did* start successfully.
        assert controller.state.rtlsdr_streaming
        # HackRF did not start.
        assert not controller.state.hackrf_streaming

        # Clean shutdown should still work.
        controller.stop_all()
        assert not controller.state.rtlsdr_streaming


# ===========================================================================
# 2. Buffer overrun handling
# ===========================================================================


class TestBufferOverrunHandling:
    """Verify the controller does not crash when the buffer overflows."""

    def test_drop_oldest_policy_no_crash(self):
        """With the default DROP_OLDEST policy, writing more samples
        than the buffer capacity should not crash.
        """
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()

        buf = controller.rtlsdr_buffer
        capacity = buf.capacity

        # Write 1.5x capacity through the device callback.
        chunk = _make_samples(capacity)
        controller._rtlsdr._rx_callback(chunk)
        extra = _make_samples(capacity // 2)
        controller._rtlsdr._rx_callback(extra)

        # Buffer should be at most full -- not crash.
        assert buf.available <= capacity
        # Overflow counter should be incremented.
        assert buf.stats.overflow_count > 0

        controller.stop_all()

    def test_drop_newest_policy_no_crash(self):
        """With DROP_NEWEST policy, samples exceeding capacity are
        silently discarded without crashing.
        """
        from sdr_module.core.sample_buffer import BufferOverflowPolicy

        # Replace the RTL-SDR buffer with a small DROP_NEWEST buffer.
        controller = DualSDRController()
        small_buf = SampleBuffer(
            capacity=1024, overflow_policy=BufferOverflowPolicy.DROP_NEWEST
        )
        controller._rtlsdr_buffer = small_buf
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()

        # Fill the buffer.
        controller._rtlsdr._rx_callback(_make_samples(1024))
        assert small_buf.available == 1024

        # Write more -- these should be dropped (DROP_NEWEST).
        controller._rtlsdr._rx_callback(_make_samples(512))
        # Buffer should not exceed capacity.
        assert small_buf.available <= 1024
        assert small_buf.stats.overflow_count > 0
        assert small_buf.stats.samples_dropped >= 512

        controller.stop_all()

    def test_rapid_writes_do_not_corrupt_buffer(self):
        """Hammer the buffer from multiple threads to verify no corruption."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()

        errors = []

        def _writer():
            try:
                for _ in range(100):
                    controller._rtlsdr._rx_callback(_make_samples(256))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Errors during concurrent writes: {errors}"

        # Buffer invariant: available <= capacity.
        assert controller.rtlsdr_buffer.available <= controller.rtlsdr_buffer.capacity

        controller.stop_all()


# ===========================================================================
# 3. Mode switching during active streaming
# ===========================================================================


class TestModeSwitchDuringActiveStreaming:
    """Verify mode switch behaviour while devices are streaming."""

    def test_set_mode_rejected_while_streaming(self):
        """set_mode should return False if any device is streaming."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()
        assert controller.state.rtlsdr_streaming

        result = controller.set_mode(OperationMode.FULL_DUPLEX)
        assert not result
        # Mode should remain DUAL_RX.
        assert controller.state.mode == OperationMode.DUAL_RX

        controller.stop_all()

    def test_stop_then_switch_mode_succeeds(self):
        """After stopping, the controller should accept a mode switch."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()
        controller.stop_all()

        result = controller.set_mode(OperationMode.FULL_DUPLEX)
        assert result
        assert controller.state.mode == OperationMode.FULL_DUPLEX

    def test_dual_rx_to_full_duplex_transition(self):
        """Start dual_rx, stop cleanly, switch to full_duplex, start again."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        # Phase 1: dual RX
        assert controller.start_dual_rx()
        assert controller.state.mode == OperationMode.DUAL_RX
        assert controller.state.rtlsdr_streaming
        assert controller.state.hackrf_streaming

        controller.stop_all()
        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming

        # Phase 2: full duplex
        tx_gen = Mock(return_value=np.zeros(1024, dtype=np.complex64))
        assert controller.start_full_duplex(rx_callback=Mock(), tx_generator=tx_gen)
        assert controller.state.mode == OperationMode.FULL_DUPLEX
        assert controller.state.rtlsdr_streaming
        assert controller.state.hackrf_transmitting

        controller.stop_all()
        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_transmitting

    def test_mode_switch_preserves_frequency_settings(self):
        """Frequencies should survive a mode switch."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.set_rtlsdr_frequency(144.5e6)
        controller.set_hackrf_frequency(433e6)

        controller.set_mode(OperationMode.FULL_DUPLEX)

        assert controller.state.rtlsdr_frequency == 144.5e6
        assert controller.state.hackrf_frequency == 433e6


# ===========================================================================
# 4. Initialize with no devices available
# ===========================================================================


class TestInitializeWithNoDevices:
    """Verify initialize() returns False when no hardware is found."""

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    def test_initialize_returns_false_no_devices(
        self, mock_get_hackrf, mock_get_rtlsdr, mock_scan
    ):
        """When scan_devices finds nothing and get_* return None,
        initialize() should return False.
        """
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
    def test_state_clean_after_failed_init(
        self, mock_get_hackrf, mock_get_rtlsdr, mock_scan
    ):
        """State should remain at defaults after a failed initialize()."""
        mock_scan.return_value = []
        mock_get_rtlsdr.return_value = None
        mock_get_hackrf.return_value = None

        controller = DualSDRController()
        controller.initialize()

        state = controller.state
        assert state.mode == OperationMode.DUAL_RX
        assert not state.rtlsdr_streaming
        assert not state.hackrf_streaming
        assert not state.hackrf_transmitting
        assert state.rtlsdr_frequency == 0.0
        assert state.hackrf_frequency == 0.0

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    def test_operations_safe_after_failed_init(
        self, mock_get_hackrf, mock_get_rtlsdr, mock_scan
    ):
        """Operations after a failed init should return False, not raise."""
        mock_scan.return_value = []
        mock_get_rtlsdr.return_value = None
        mock_get_hackrf.return_value = None

        controller = DualSDRController()
        controller.initialize()

        # Frequency setting should return False (no device).
        assert not controller.set_rtlsdr_frequency(100e6)
        assert not controller.set_hackrf_frequency(433e6)

        # start_full_duplex requires both devices.
        assert not controller.start_full_duplex()

        # stop_all should be harmless.
        controller.stop_all()

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    def test_initialize_partial_rtlsdr_only(
        self, mock_get_hackrf, mock_get_rtlsdr, mock_scan
    ):
        """If only RTL-SDR is found, initialize should still return True
        (at least one device is present).
        """
        mock_scan.return_value = []
        mock_get_rtlsdr.return_value = MockRTLSDRDevice()
        mock_get_hackrf.return_value = None

        controller = DualSDRController()
        # apply_config is called for the RTL-SDR device; patch it.
        with patch.object(
            controller._device_manager, "apply_config", return_value=True
        ):
            result = controller.initialize()

        assert result
        assert controller.rtlsdr is not None
        assert controller.hackrf is None

    @patch("sdr_module.core.device_manager.DeviceManager.scan_devices")
    @patch("sdr_module.core.device_manager.DeviceManager.get_rtlsdr")
    @patch("sdr_module.core.device_manager.DeviceManager.get_hackrf")
    def test_initialize_scan_raises_no_crash(
        self, mock_get_hackrf, mock_get_rtlsdr, mock_scan
    ):
        """If scan_devices raises, initialize should propagate or handle
        gracefully without leaving the controller in a broken state.
        """
        mock_scan.side_effect = OSError("USB subsystem error")
        mock_get_rtlsdr.return_value = None
        mock_get_hackrf.return_value = None

        controller = DualSDRController()
        with pytest.raises(OSError, match="USB subsystem error"):
            controller.initialize()

        # Even after the exception, basic accessors should work.
        assert controller.rtlsdr is None
        assert controller.hackrf is None


# ===========================================================================
# 5. Double start / stop
# ===========================================================================


class TestDoubleStartStop:
    """Calling start or stop multiple times must not crash."""

    def test_double_start_dual_rx(self):
        """Calling start_dual_rx twice should not crash."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        first = controller.start_dual_rx()
        assert first

        # Second call -- the devices are already streaming but the
        # controller re-invokes start_rx.  This must not crash.
        controller.start_dual_rx()
        # Regardless of the return value, state must be consistent.
        assert controller.state.rtlsdr_streaming
        assert controller.state.hackrf_streaming

        controller.stop_all()

    def test_double_stop_all(self):
        """Calling stop_all twice should not crash."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()

        controller.stop_all()
        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming

        # Second stop -- nothing is streaming, should be a no-op.
        controller.stop_all()
        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming

    def test_stop_without_start(self):
        """stop_all on a freshly constructed controller should be safe."""
        controller = DualSDRController()
        controller.stop_all()  # Should not raise.
        assert not controller.state.rtlsdr_streaming

    def test_double_stop_rtlsdr(self):
        """Calling stop_rtlsdr twice should not crash."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()
        controller.stop_rtlsdr()
        assert not controller.state.rtlsdr_streaming

        controller.stop_rtlsdr()  # second call, should be harmless
        assert not controller.state.rtlsdr_streaming

        controller.stop_all()

    def test_double_stop_hackrf(self):
        """Calling stop_hackrf twice should not crash."""
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        controller.start_dual_rx()
        controller.stop_hackrf()
        assert not controller.state.hackrf_streaming

        controller.stop_hackrf()  # second call, should be harmless
        assert not controller.state.hackrf_streaming

        controller.stop_all()

    def test_start_after_stop_restarts_cleanly(self):
        """Starting again after a full stop should work identically
        to the first start.
        """
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()

        assert controller.start_dual_rx()
        controller.stop_all()

        # Start again.
        assert controller.start_dual_rx()
        assert controller.state.rtlsdr_streaming
        assert controller.state.hackrf_streaming

        # Feed samples -- should still write to buffer.
        controller._rtlsdr._rx_callback(_make_samples(100))
        assert controller.rtlsdr_buffer.available > 0

        controller.stop_all()


# ===========================================================================
# 6. Frequency setting on uninitialized controller
# ===========================================================================


class TestFrequencyOnUninitializedController:
    """Verify set_rtlsdr_frequency and set_hackrf_frequency return False
    when no device is connected.
    """

    def test_set_rtlsdr_frequency_no_device(self):
        """set_rtlsdr_frequency must return False with no RTL-SDR."""
        controller = DualSDRController()
        # No devices assigned.
        result = controller.set_rtlsdr_frequency(100e6)
        assert not result

    def test_set_hackrf_frequency_no_device(self):
        """set_hackrf_frequency must return False with no HackRF."""
        controller = DualSDRController()
        result = controller.set_hackrf_frequency(433e6)
        assert not result

    def test_set_frequencies_no_devices(self):
        """set_frequencies must return (False, False) with no devices."""
        controller = DualSDRController()
        rtl_ok, hack_ok = controller.set_frequencies(100e6, 433e6)
        assert not rtl_ok
        assert not hack_ok

    def test_frequency_not_updated_in_state_on_failure(self):
        """State frequencies should remain 0.0 after a failed set."""
        controller = DualSDRController()
        controller.set_rtlsdr_frequency(144e6)
        controller.set_hackrf_frequency(433e6)

        assert controller.state.rtlsdr_frequency == 0.0
        assert controller.state.hackrf_frequency == 0.0

    def test_set_rtlsdr_frequency_only_rtlsdr_present(self):
        """set_rtlsdr_frequency should succeed when RTL-SDR is present,
        even without HackRF.
        """
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        # No HackRF.

        assert controller.set_rtlsdr_frequency(144.5e6)
        assert controller.state.rtlsdr_frequency == 144.5e6

        # HackRF frequency should fail.
        assert not controller.set_hackrf_frequency(433e6)

    def test_set_hackrf_frequency_only_hackrf_present(self):
        """set_hackrf_frequency should succeed when HackRF is present,
        even without RTL-SDR.
        """
        controller = DualSDRController()
        controller._hackrf = MockHackRFDevice()
        # No RTL-SDR.

        assert controller.set_hackrf_frequency(915e6)
        assert controller.state.hackrf_frequency == 915e6

        # RTL-SDR frequency should fail.
        assert not controller.set_rtlsdr_frequency(100e6)

    def test_get_status_without_devices(self):
        """get_status should work even with no devices connected."""
        controller = DualSDRController()
        status = controller.get_status()

        assert status["rtlsdr"]["connected"] is False
        assert status["hackrf"]["connected"] is False
        assert status["mode"] == OperationMode.DUAL_RX.value

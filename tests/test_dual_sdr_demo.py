"""Dual-SDR without hardware (FG-B16): the controller runs on two demo
devices through :class:`~sdr_module.devices.polled.PolledDevice`.

Before, the demo device didn't have the SDRDevice streaming API (no
``start_rx(callback)``, no ``state``), so ``start_dual_rx`` with demo devices
raised ``TypeError: MockDevice.start_rx() takes 1 positional argument but 2
were given``, and the controller had no public way to be given devices.
"""

import threading
import time

import numpy as np
import pytest

from sdr_module.core.dual_sdr import DualSDRController, OperationMode
from sdr_module.devices.base import SDRDevice
from sdr_module.devices.polled import PolledDevice
from sdr_module.gui.device_dialog import MockDevice


class Collector:
    """A thread-safe sample callback."""

    def __init__(self):
        self.count = 0
        self.blocks = 0
        self.dtype = None
        self._lock = threading.Lock()

    def __call__(self, samples):
        with self._lock:
            self.count += len(samples)
            self.blocks += 1
            self.dtype = samples.dtype


def wait_for(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def demo(name="Demo", freq=100.1e6, rate=2.4e6):
    device = MockDevice()
    device.set_sample_rate(rate)
    device.set_frequency(freq)
    return PolledDevice(device, name)


class TestPolledDevice:
    def test_is_a_receive_only_sdr_device_mirroring_the_source(self):
        device = demo("Demo A", 433.92e6)
        assert isinstance(device, SDRDevice)
        assert device.info.name == "Demo A"
        assert device.can_receive() and not device.can_transmit()
        assert device.state.frequency == 433.92e6
        assert device.state.sample_rate == 2.4e6
        assert device.set_frequency(145.8e6)
        assert device.source.frequency == 145.8e6
        assert device.state.frequency == 145.8e6
        assert device.set_gain(30)
        assert device.state.gain == 30
        assert device.set_gain_mode(True)
        assert device.state.gain_mode == "auto"
        with pytest.raises(NotImplementedError):
            device.start_tx(lambda: None)

    def test_streams_in_real_time_to_a_callback(self):
        device = demo()
        got = Collector()
        assert device.start_rx(got)
        assert device.state.is_streaming
        started = time.monotonic()
        assert wait_for(lambda: got.count >= 0.25 * 2.4e6)
        elapsed = time.monotonic() - started
        device.stop_rx()
        assert not device.state.is_streaming
        # About real time (not much faster than the clock).
        assert got.count <= 2.4e6 * (elapsed + 0.1)
        assert got.dtype == np.complex64
        after = got.count
        time.sleep(0.1)
        assert got.count == after  # (stopped)

    def test_queues_for_read_samples_without_a_callback(self):
        device = demo()
        assert device.start_rx()
        block = device.read_samples(4096, timeout=2.0)
        device.stop_rx()
        assert block is not None and len(block) > 0

    def test_a_failing_source_ends_the_stream_with_its_reason(self):
        class Broken:
            sample_rate = 1e6

            def start_rx(self):
                return True

            def stop_rx(self):
                return True

            def read_samples(self, n):
                raise OSError("unplugged")

        device = PolledDevice(Broken(), "Broken")
        assert device.start_rx(Collector())
        assert wait_for(lambda: not device.state.is_streaming)
        assert device.rx_error == "unplugged"
        device.stop_rx()

    def test_close_closes_the_source(self):
        device = demo()
        device.start_rx(Collector())
        device.close()
        assert not device.state.is_streaming
        assert not device.source.is_streaming
        assert not device.is_open


class TestControllerOnDemoDevices:
    def test_dual_receive_on_two_demo_devices(self):
        controller = DualSDRController()
        assert controller.use_devices(demo("A", 433.92e6), demo("B", 915e6))
        assert controller.state.rtlsdr_frequency == 433.92e6
        assert controller.state.hackrf_frequency == 915e6
        first, second = Collector(), Collector()
        assert controller.start_dual_rx(first, second)
        state = controller.state
        assert state.mode == OperationMode.DUAL_RX
        assert state.rtlsdr_streaming and state.hackrf_streaming
        assert wait_for(lambda: first.count > 100_000 and second.count > 100_000)
        assert controller.rtlsdr_buffer.available > 0
        assert controller.hackrf_buffer.available > 0
        samples = controller.read_rtlsdr_samples(4096, timeout=1.0)
        assert samples is not None and len(samples) == 4096
        assert controller.set_rtlsdr_frequency(434.0e6)
        assert controller.rtlsdr.source.frequency == 434.0e6
        controller.stop_all()
        assert not controller.state.rtlsdr_streaming
        assert not controller.state.hackrf_streaming
        assert not controller.rtlsdr.state.is_streaming
        assert not controller.hackrf.state.is_streaming

    def test_full_duplex_needs_a_device_that_can_transmit(self):
        controller = DualSDRController()
        controller.use_devices(demo("A"), demo("B"))
        tx = lambda: np.zeros(1024, dtype=np.complex64)  # noqa: E731
        assert not controller.start_full_duplex(Collector(), tx)
        assert not controller.state.hackrf_transmitting
        assert not controller.state.rtlsdr_streaming
        assert not controller.start_tx_monitor(tx, Collector())

    def test_devices_cant_change_while_streaming(self):
        controller = DualSDRController()
        controller.use_devices(demo("A"))
        controller.start_dual_rx(Collector())
        assert not controller.use_devices(demo("B"), demo("C"))
        assert controller.rtlsdr.info.name == "A"
        controller.stop_all()
        assert controller.use_devices(None, demo("C"))
        assert controller.rtlsdr is None
        assert controller.hackrf.info.name == "C"
        assert not controller.use_devices()

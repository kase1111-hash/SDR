"""Unit tests for the RTL-SDR driver against a fake ``rtlsdr`` backend.

The fake mirrors the real pyrtlsdr surface: ``RtlSdr`` instances expose
``sample_rate`` / ``center_freq`` / ``gain`` properties, ``read_samples``,
``close``, ``set_bias_tee``, ``set_direct_sampling``; enumeration goes through
``RtlSdr.get_device_serial_addresses`` and ``librtlsdr.rtlsdr_get_device_count``.
The old driver called ``RtlSdr.get_device_count`` / ``get_device_serial``,
which pyrtlsdr never had; those names are intentionally absent here so a
regression would raise AttributeError.
"""

import sys
import types

import numpy as np
import pytest

from sdr_module.devices.rtlsdr import RTLSDRDevice


class FakeRtlSdr:
    """Fake pyrtlsdr RtlSdr instance."""

    _serials: list = []

    def __init__(self, device_index=0):
        self.device_index = device_index
        self._sample_rate = 2.048e6
        self._center_freq = 100e6
        self._gain = "auto"
        self.closed = False
        self.bias_tee = None
        self.direct_sampling = None
        self._read_data = np.zeros(1024, dtype=np.complex128)

    # Enumeration (real static API).
    @staticmethod
    def get_device_serial_addresses():
        return list(FakeRtlSdr._serials)

    @staticmethod
    def get_device_index_by_serial(serial):
        return FakeRtlSdr._serials.index(serial)

    # Properties.
    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, v):
        self._sample_rate = v

    @property
    def center_freq(self):
        return self._center_freq

    @center_freq.setter
    def center_freq(self, v):
        self._center_freq = v

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, v):
        self._gain = v

    def read_samples(self, n):
        return self._read_data[:n]

    def set_bias_tee(self, enabled):
        self.bias_tee = enabled

    def set_direct_sampling(self, mode):
        self.direct_sampling = mode

    def close(self):
        self.closed = True


def make_fake_backend(serials=("77771111",), device_count=None):
    rtlsdr = types.ModuleType("rtlsdr")
    librtlsdr = types.ModuleType("rtlsdr.librtlsdr")
    FakeRtlSdr._serials = list(serials)
    if device_count is None:
        # Track the live serial list so a test that changes it after setup
        # still sees a consistent count.
        librtlsdr.rtlsdr_get_device_count = lambda: len(FakeRtlSdr._serials)
    else:
        librtlsdr.rtlsdr_get_device_count = lambda: device_count
    rtlsdr.RtlSdr = FakeRtlSdr
    rtlsdr.librtlsdr = librtlsdr
    return rtlsdr, librtlsdr


@pytest.fixture
def fake_backend(monkeypatch):
    rtlsdr, librtlsdr = make_fake_backend()
    monkeypatch.setitem(sys.modules, "rtlsdr", rtlsdr)
    monkeypatch.setitem(sys.modules, "rtlsdr.librtlsdr", librtlsdr)
    return rtlsdr


# --------------------------------------------------------------------------- #
# Missing library
# --------------------------------------------------------------------------- #
def test_list_devices_without_library(monkeypatch):
    # Make the lazy `from rtlsdr import ...` fail.
    monkeypatch.setitem(sys.modules, "rtlsdr", None)
    assert RTLSDRDevice.list_devices() == []
    assert RTLSDRDevice.get_device_count() == 0


def test_open_without_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtlsdr", None)
    d = RTLSDRDevice()
    assert d.open() is False
    assert d.is_open is False


# --------------------------------------------------------------------------- #
# Enumeration uses the REAL pyrtlsdr API
# --------------------------------------------------------------------------- #
def test_list_devices(fake_backend):
    FakeRtlSdr._serials = ["AAAA", "BBBB"]
    infos = RTLSDRDevice.list_devices()
    assert [i.serial for i in infos] == ["AAAA", "BBBB"]
    assert RTLSDRDevice.get_device_count() == 2
    assert RTLSDRDevice.get_device_serial(1) == "BBBB"
    assert RTLSDRDevice.get_device_serial(5) is None


def test_list_devices_zero(monkeypatch):
    rtlsdr, librtlsdr = make_fake_backend(serials=())
    monkeypatch.setitem(sys.modules, "rtlsdr", rtlsdr)
    monkeypatch.setitem(sys.modules, "rtlsdr.librtlsdr", librtlsdr)
    assert RTLSDRDevice.list_devices() == []


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_open_and_defaults(fake_backend):
    d = RTLSDRDevice()
    assert d.open() is True
    assert d.is_open is True
    assert d.state.sample_rate == 2.4e6
    assert d.state.frequency == 100e6
    d.close()
    assert d.is_open is False


def test_open_failure_does_not_leak(fake_backend, monkeypatch):
    # Make a default setter raise after construction succeeds.
    def bad_sr(self, v):
        raise OSError("rtlsdr_set_sample_rate failed")

    monkeypatch.setattr(FakeRtlSdr, "sample_rate", property(lambda self: 0, bad_sr))
    d = RTLSDRDevice()
    assert d.open() is False
    # Must not report open, and the handle must have been closed.
    assert d.is_open is False
    assert d._device is None


def test_set_frequency_out_of_range(fake_backend):
    d = RTLSDRDevice()
    d.open()
    assert d.set_frequency(3e9) is False


def test_set_gain_rounds_to_valid(fake_backend):
    d = RTLSDRDevice()
    d.open()
    assert d.set_gain(30.0) is True
    # 30 dB rounds to the nearest valid R820T2 step (29.7).
    assert d.state.gain == 29.7


# --------------------------------------------------------------------------- #
# RX
# --------------------------------------------------------------------------- #
def test_rx_queues_samples(fake_backend):
    d = RTLSDRDevice()
    d.open()
    d._device._read_data = np.ones(4096, dtype=np.complex128)
    assert d.start_rx() is True
    samples = d.read_samples(4096, timeout=2.0)
    assert samples is not None
    assert len(samples) > 0
    d.stop_rx()
    assert d.state.is_streaming is False


def test_rx_thread_error_surfaced(fake_backend):
    d = RTLSDRDevice()
    d.open()

    def bad_read(_n):
        raise OSError("usb fell out")

    d._device.read_samples = bad_read
    d.start_rx()
    # Give the RX thread a moment to hit the error and clear the flag.
    for _ in range(200):
        if not d.state.is_streaming:
            break
        import time

        time.sleep(0.01)
    assert d.state.is_streaming is False
    assert "usb fell out" in (d.rx_error or "")

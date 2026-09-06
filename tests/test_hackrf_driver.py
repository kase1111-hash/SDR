"""Unit tests for the HackRF driver against a fake ``python_hackrf`` backend.

The fake deliberately mirrors the *real* ``python_hackrf`` public surface
(method names, the four-argument RX/TX callback contract, exceptions instead
of return codes). Attribute access outside that allow-list raises
AttributeError, so a driver call to a method the real library does not have
fails the test instead of silently passing against an over-permissive Mock.
"""

import sys
import types

import numpy as np
import pytest

import sdr_module.devices.hackrf as hackrf_mod
from sdr_module.devices.hackrf import HackRFDevice

# Real python_hackrf device method names (from pyhackrf.pyi), the only
# attributes the fake device is allowed to expose.
_ALLOWED_DEVICE_METHODS = {
    "pyhackrf_close",
    "pyhackrf_serialno_read",
    "pyhackrf_set_freq",
    "pyhackrf_set_sample_rate",
    "pyhackrf_set_baseband_filter_bandwidth",
    "pyhackrf_set_lna_gain",
    "pyhackrf_set_vga_gain",
    "pyhackrf_set_txvga_gain",
    "pyhackrf_set_amp_enable",
    "pyhackrf_is_streaming",
    "set_rx_callback",
    "set_tx_callback",
    "pyhackrf_start_rx",
    "pyhackrf_stop_rx",
    "pyhackrf_start_tx",
    "pyhackrf_stop_tx",
}


class _CInt:
    """Stand-in for the ctypes c_int the real TX callback receives."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


class FakeHackRFDevice:
    """Fake device that only answers to real python_hackrf method names."""

    def __init__(self, serial="0000000000abcdef", buffer_length=262144):
        self._serial = serial
        self._buffer_length = buffer_length
        self.calls: list = []
        self._rx_cb = None
        self._tx_cb = None
        self.closed = False
        self.freq = None
        self.sample_rate = None
        self.bandwidth = None
        self.lna = None
        self.vga = None
        self.txvga = None
        self.amp = None
        # When True, pyhackrf_start_tx immediately drives the TX callback,
        # mimicking libhackrf's USB thread pulling samples synchronously.
        self.auto_pump = False
        self.tx_output = np.empty(0, dtype=np.int8)

    def __getattr__(self, name):
        # Anything not defined below and not in the allow-list is a bug: the
        # real library would not have it either.
        if name not in _ALLOWED_DEVICE_METHODS:
            raise AttributeError(f"python_hackrf device has no attribute {name!r}")
        raise AttributeError(name)

    def pyhackrf_serialno_read(self):
        return self._serial

    def pyhackrf_set_freq(self, freq):
        self.calls.append(("set_freq", freq))
        self.freq = freq

    def pyhackrf_set_sample_rate(self, rate):
        self.calls.append(("set_sample_rate", rate))
        self.sample_rate = rate

    def pyhackrf_set_baseband_filter_bandwidth(self, bw):
        self.calls.append(("set_bw", bw))
        self.bandwidth = bw

    def pyhackrf_set_lna_gain(self, g):
        self.lna = g

    def pyhackrf_set_vga_gain(self, g):
        self.vga = g

    def pyhackrf_set_txvga_gain(self, g):
        self.txvga = g

    def pyhackrf_set_amp_enable(self, on):
        self.amp = on

    def set_rx_callback(self, fn):
        self._rx_cb = fn

    def set_tx_callback(self, fn):
        self._tx_cb = fn

    def pyhackrf_start_rx(self):
        self.calls.append(("start_rx",))

    def pyhackrf_stop_rx(self):
        self.calls.append(("stop_rx",))

    def pyhackrf_start_tx(self):
        self.calls.append(("start_tx",))
        if self.auto_pump and self._tx_cb is not None:
            self.tx_output = self.pump_tx()

    def pyhackrf_stop_tx(self):
        self.calls.append(("stop_tx",))

    def pyhackrf_close(self):
        self.closed = True

    # --- test helpers (not part of the real API) ---
    def feed_rx(self, int8_bytes, valid_length=None):
        """Deliver one RX transfer to the registered callback."""
        buf = np.zeros(self._buffer_length, dtype=np.int8)
        buf[: len(int8_bytes)] = int8_bytes
        vl = len(int8_bytes) if valid_length is None else valid_length
        return self._rx_cb(self, buf, self._buffer_length, vl)

    def pump_tx(self, transfers=100):
        """Drive the TX callback until it asks to stop, collecting bytes."""
        out = bytearray()
        for _ in range(transfers):
            buf = np.zeros(self._buffer_length, dtype=np.int8)
            vl = _CInt(0)
            rc = self._tx_cb(self, buf, self._buffer_length, vl)
            out.extend(bytes(buf[: vl.value].tobytes()))
            if rc != 0:
                break
        return np.frombuffer(bytes(out), dtype=np.int8)


class FakeDeviceList:
    def __init__(self, serials):
        self.serial_numbers = list(serials)
        self.device_count = len(serials)


def make_fake_backend(serials=("0000000000abcdef",), device=None):
    """Build a fake ``python_hackrf`` module exposing the real entry points."""
    mod = types.ModuleType("python_hackrf")
    pyhackrf = types.ModuleType("python_hackrf.pyhackrf")
    state = {
        "device": device or FakeHackRFDevice(serial=serials[0] if serials else "x")
    }

    pyhackrf.pyhackrf_init = lambda: None
    pyhackrf.pyhackrf_exit = lambda: None
    pyhackrf.pyhackrf_device_list = lambda: FakeDeviceList(serials)
    pyhackrf.pyhackrf_open = lambda: state["device"] if serials else None
    pyhackrf.pyhackrf_device_list_open = lambda dl, i: state["device"]
    pyhackrf.pyhackrf_compute_baseband_filter_bw = lambda bw: bw
    mod.pyhackrf = pyhackrf
    return mod, pyhackrf, state["device"]


@pytest.fixture
def fake_backend(monkeypatch):
    mod, pyhackrf, device = make_fake_backend()
    monkeypatch.setitem(sys.modules, "python_hackrf", mod)
    monkeypatch.setitem(sys.modules, "python_hackrf.pyhackrf", pyhackrf)
    # Force re-init each test so the once-per-process guard does not leak.
    monkeypatch.setattr(hackrf_mod, "_initialized", False)
    return device


# --------------------------------------------------------------------------- #
# Missing-library behaviour
# --------------------------------------------------------------------------- #
def test_list_devices_without_library(monkeypatch):
    monkeypatch.setattr(hackrf_mod, "_load_backend", lambda: None)
    assert HackRFDevice.list_devices() == []


def test_open_without_library(monkeypatch):
    monkeypatch.setattr(hackrf_mod, "_load_backend", lambda: None)
    d = HackRFDevice()
    assert d.open() is False
    assert d.is_open is False


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def test_list_devices_zero(monkeypatch):
    mod, pyhackrf, _ = make_fake_backend(serials=())
    monkeypatch.setitem(sys.modules, "python_hackrf", mod)
    monkeypatch.setitem(sys.modules, "python_hackrf.pyhackrf", pyhackrf)
    monkeypatch.setattr(hackrf_mod, "_initialized", False)
    assert HackRFDevice.list_devices() == []


def test_list_devices_two(monkeypatch):
    mod, pyhackrf, _ = make_fake_backend(serials=("aaaa", "bbbb"))
    monkeypatch.setitem(sys.modules, "python_hackrf", mod)
    monkeypatch.setitem(sys.modules, "python_hackrf.pyhackrf", pyhackrf)
    monkeypatch.setattr(hackrf_mod, "_initialized", False)
    infos = HackRFDevice.list_devices()
    assert [i.serial for i in infos] == ["aaaa", "bbbb"]
    assert infos[0].index == 0 and infos[1].index == 1


# --------------------------------------------------------------------------- #
# Lifecycle and tuning
# --------------------------------------------------------------------------- #
def test_open_sets_defaults_and_closes(fake_backend):
    d = HackRFDevice()
    assert d.open() is True
    assert d.is_open is True
    assert d.info.serial == "0000000000abcdef"
    # Defaults were pushed to the device.
    assert fake_backend.sample_rate == 10e6
    assert fake_backend.freq == int(100e6)
    d.close()
    assert fake_backend.closed is True
    assert d.is_open is False


def test_set_frequency_out_of_range(fake_backend):
    d = HackRFDevice()
    d.open()
    assert d.set_frequency(50e9) is False


def test_set_bandwidth_rounds_to_supported(fake_backend):
    d = HackRFDevice()
    d.open()
    assert d.set_bandwidth(9.3e6) is True
    assert d.state.bandwidth == 9e6


# --------------------------------------------------------------------------- #
# RX callback conversion
# --------------------------------------------------------------------------- #
def test_rx_callback_converts_int8_to_complex(fake_backend):
    d = HackRFDevice()
    d.open()
    assert d.start_rx() is True
    # Interleaved I/Q: (127, 0), (-128, 127)
    fake_backend.feed_rx(np.array([127, 0, -128, 127], dtype=np.int8))
    samples = d.read_samples(2, timeout=1.0)
    assert samples is not None
    assert samples.dtype == np.complex64
    assert np.isclose(samples[0].real, 127 / 128.0)
    assert np.isclose(samples[0].imag, 0.0)
    assert np.isclose(samples[1].real, -1.0)
    assert np.isclose(samples[1].imag, 127 / 128.0)
    d.stop_rx()
    assert d.state.is_streaming is False


def test_rx_callback_only_uses_valid_length(fake_backend):
    d = HackRFDevice()
    d.open()
    d.start_rx()
    # Provide 8 real bytes but claim only 4 are valid -> 2 samples.
    fake_backend.feed_rx(
        np.array([10, 20, 30, 40, 99, 99, 99, 99], dtype=np.int8), valid_length=4
    )
    samples = d.read_samples(2, timeout=1.0)
    assert len(samples) == 2
    d.stop_rx()


def test_rx_callback_error_surfaced(fake_backend):
    d = HackRFDevice()
    d.open()

    def boom(_samples):
        raise ValueError("consumer exploded")

    d.start_rx(callback=boom)
    rc = fake_backend.feed_rx(np.array([1, 2, 3, 4], dtype=np.int8))
    assert rc != 0  # callback asked libhackrf to stop
    assert d.state.is_streaming is False
    assert "consumer exploded" in (d.rx_error or "")


# --------------------------------------------------------------------------- #
# TX
# --------------------------------------------------------------------------- #
def test_start_tx_requires_callback(fake_backend):
    d = HackRFDevice()
    d.open()
    d.set_frequency(100e6)
    assert d.start_tx(None) is False
    assert d.state.is_transmitting is False


def test_tx_callback_clips_and_chunks(fake_backend, monkeypatch):
    # This test exercises buffer conversion, not the lockout logic (covered by
    # the dedicated blocked-frequency tests), so allow TX unconditionally here.
    monkeypatch.setattr(hackrf_mod, "is_tx_allowed", lambda *a, **k: (True, None))
    d = HackRFDevice()
    d.open()
    d.set_frequency(100e6)
    blocks = [np.array([1.5 - 1.5j, 0.5 + 0.25j], dtype=np.complex64)]

    def gen():
        return blocks.pop(0) if blocks else None

    assert d.start_tx(gen) is True
    out = fake_backend.pump_tx()
    # 2 complex samples -> 4 int8, clipped to +/-127.
    assert out[0] == 127  # 1.5 clipped -> 1.0 -> 127
    assert out[1] == -127
    assert out[2] == round(0.5 * 127)
    assert out[3] == round(0.25 * 127)
    d.stop_tx()


def test_write_samples_blocks_and_transmits(fake_backend, monkeypatch):
    monkeypatch.setattr(hackrf_mod, "is_tx_allowed", lambda *a, **k: (True, None))
    d = HackRFDevice()
    d.open()
    d.set_frequency(100e6)
    # Auto-pump makes pyhackrf_start_tx drive the callback synchronously, the
    # way libhackrf's USB thread would, so write_samples' wait resolves.
    fake_backend.auto_pump = True
    samples = np.array([0.5 + 0.5j, -0.5 - 0.5j], dtype=np.complex64)
    result = d.write_samples(samples, timeout=5.0)
    assert result is True
    assert d.state.is_transmitting is False
    # The whole block was handed to the device (2 samples -> 4 int8 bytes).
    assert len(fake_backend.tx_output) == 4


def test_tx_blocked_on_locked_frequency(fake_backend):
    d = HackRFDevice()
    d.open()
    # 1090 MHz is an ADS-B lockout; TX must be refused.
    d.set_frequency(1090e6)
    assert d.start_tx(lambda: np.zeros(4, dtype=np.complex64)) is False
    assert d.write_samples(np.ones(4, dtype=np.complex64)) is False


def test_gps_frequency_tx_blocked(fake_backend):
    d = HackRFDevice()
    d.open()
    d.set_frequency(1575.42e6)  # GPS L1
    assert d.start_tx(lambda: np.zeros(4, dtype=np.complex64)) is False


def test_half_duplex_rx_blocks_tx(fake_backend):
    d = HackRFDevice()
    d.open()
    d.set_frequency(100e6)
    d.start_rx()
    assert d.start_tx(lambda: np.zeros(4, dtype=np.complex64)) is False
    d.stop_rx()

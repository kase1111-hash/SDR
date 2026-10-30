"""Device front-end settings in the drivers (T11): the RTL-SDR's own gain
table, frequency correction (ppm), direct sampling and bias tee; the
HackRF One's missing AGC, antenna power (bias tee) and LNA/VGA readback.

No hardware: the fakes mirror the real libraries' surfaces (pyrtlsdr 0.5's
``RtlSdr``, python_hackrf 1.5's ``PyHackrfDevice``), including librtlsdr's
refusal to set the frequency correction it already has (pyrtlsdr then
closes the device).
"""

import logging
import sys
import types

import pytest

import sdr_module.devices.hackrf as hackrf_mod
from sdr_module.devices.base import DeviceCapability
from sdr_module.devices.hackrf import HackRFDevice
from sdr_module.devices.rtlsdr import (
    DIRECT_SAMPLING_I,
    DIRECT_SAMPLING_OFF,
    DIRECT_SAMPLING_Q,
    RTLSDRDevice,
    direct_sampling_mode,
)

E4000_GAINS = [-1.0, 1.5, 4.0, 6.5, 9.0, 11.5, 14.0, 16.5, 19.0, 21.5, 24.0, 29.0]


class FakeRtlSdr:
    """pyrtlsdr's RtlSdr: the attributes and methods the driver may use."""

    gains_db = None  # valid_gains_db of the next device opened (None: R820T)

    def __init__(self, device_index=0):
        self.device_index = device_index
        self.sample_rate = 2.048e6
        self.center_freq = 100e6
        self.gain = "auto"
        self.calls = []
        self.corr = 0
        self.direct = 0
        self.bias = False
        self.closed = False
        if FakeRtlSdr.gains_db is not None:
            self.valid_gains_db = list(FakeRtlSdr.gains_db)
        else:
            self.valid_gains_db = list(RTLSDRDevice.VALID_GAINS)

    @staticmethod
    def get_device_serial_addresses():
        return ["00000001"]

    def set_freq_correction(self, ppm):
        self.calls.append(("ppm", ppm))
        if ppm == self.corr:
            # librtlsdr returns -2 for the value it has; pyrtlsdr closes.
            self.closed = True
            raise OSError("Could not set freq. offset to %d ppm" % ppm)
        self.corr = ppm

    def get_freq_correction(self):
        return self.corr

    def set_direct_sampling(self, direct):
        self.calls.append(("direct", direct))
        self.direct = direct

    def set_bias_tee(self, enabled):
        self.calls.append(("bias", bool(enabled)))
        self.bias = bool(enabled)

    def read_samples(self, n):
        import numpy as np

        return np.zeros(n, dtype=np.complex64)

    def close(self):
        self.closed = True


@pytest.fixture
def rtl(monkeypatch):
    rtlsdr = types.ModuleType("rtlsdr")
    librtlsdr = types.ModuleType("rtlsdr.librtlsdr")
    librtlsdr.rtlsdr_get_device_count = lambda: 1
    rtlsdr.RtlSdr = FakeRtlSdr
    rtlsdr.librtlsdr = librtlsdr
    monkeypatch.setitem(sys.modules, "rtlsdr", rtlsdr)
    monkeypatch.setitem(sys.modules, "rtlsdr.librtlsdr", librtlsdr)
    monkeypatch.setattr(FakeRtlSdr, "gains_db", None)
    dev = RTLSDRDevice()
    assert dev.open(0)
    yield dev
    dev.close()


# --------------------------------------------------------------------------- #
# RTL-SDR gain table
# --------------------------------------------------------------------------- #
def test_gain_table_is_the_open_tuners_own(monkeypatch, rtl):
    rtl.close()
    monkeypatch.setattr(FakeRtlSdr, "gains_db", E4000_GAINS)
    dev = RTLSDRDevice()
    assert dev.open(0)
    assert dev.get_tuner_gains() == E4000_GAINS
    assert dev.set_gain(30.0)
    assert dev.state.gain == 29.0  # the E4000's step, not the R820T's 29.7
    dev.close()


def test_gain_table_falls_back_to_the_r820t(rtl):
    rtl._device.valid_gains_db = None
    assert rtl._read_gain_table() == RTLSDRDevice.VALID_GAINS
    assert RTLSDRDevice().get_tuner_gains() == RTLSDRDevice.VALID_GAINS


def test_a_gain_between_steps_is_not_logged_as_a_problem(rtl, caplog):
    with caplog.at_level(logging.DEBUG, logger="sdr_module.devices.rtlsdr"):
        assert rtl.set_gain(31.0)
    assert rtl.state.gain == 29.7
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------- #
# RTL-SDR frequency correction
# --------------------------------------------------------------------------- #
def test_freq_correction_is_applied_and_reported(rtl):
    assert rtl.has_capability(DeviceCapability.FREQ_CORRECTION)
    assert rtl.freq_correction == 0
    assert rtl.set_freq_correction(12)
    assert rtl._device.corr == 12
    assert rtl.freq_correction == 12
    assert rtl.set_freq_correction(-3.4)  # rounded to whole ppm
    assert rtl._device.corr == -3
    assert not rtl._device.closed


def test_an_unchanged_correction_is_never_sent_again(rtl):
    """librtlsdr refuses the value it has and pyrtlsdr then closes the
    device: 0 on a fresh handle, or the same value twice, must not be sent."""
    assert rtl.set_freq_correction(0)
    assert rtl.set_freq_correction(7)
    assert rtl.set_freq_correction(7)
    assert [c for c in rtl._device.calls if c[0] == "ppm"] == [("ppm", 7)]
    assert not rtl._device.closed


def test_a_refused_correction_is_reported(rtl):
    def refuse(ppm):
        raise OSError("usb error")

    rtl._device.set_freq_correction = refuse
    assert rtl.set_freq_correction(5) is False
    assert rtl.freq_correction == 0
    assert rtl.set_freq_correction(9999) is False  # out of range
    assert RTLSDRDevice().set_freq_correction(5) is False  # not open


def test_reopening_starts_uncorrected(rtl):
    assert rtl.set_freq_correction(20)
    rtl.close()
    assert rtl.open(0)
    assert rtl.freq_correction == 0


# --------------------------------------------------------------------------- #
# RTL-SDR direct sampling
# --------------------------------------------------------------------------- #
def test_direct_sampling_mode_names():
    assert direct_sampling_mode("q") == DIRECT_SAMPLING_Q
    assert direct_sampling_mode("I") == DIRECT_SAMPLING_I
    assert direct_sampling_mode("off") == DIRECT_SAMPLING_OFF
    assert direct_sampling_mode(False) == DIRECT_SAMPLING_OFF
    assert direct_sampling_mode(2) == DIRECT_SAMPLING_Q
    with pytest.raises(ValueError):
        direct_sampling_mode("x")
    with pytest.raises(ValueError):
        direct_sampling_mode(3)


def test_direct_sampling_is_public_and_follows_the_frequency(rtl):
    assert hasattr(RTLSDRDevice, "set_direct_sampling")
    assert rtl.direct_sampling_branch == DIRECT_SAMPLING_Q  # RTL-SDR Blog V3
    assert rtl.set_frequency(7.1e6)
    assert (rtl._device.direct, rtl.direct_sampling) == (2, True)
    assert rtl.set_frequency(30e6)
    assert (rtl._device.direct, rtl.direct_sampling) == (0, False)


def test_changing_the_branch_applies_at_once_below_24_mhz(rtl):
    assert rtl.set_frequency(7.1e6)
    assert rtl.set_direct_sampling("i")
    assert rtl._device.direct == 1
    assert rtl.set_direct_sampling("off")
    assert (rtl._device.direct, rtl.direct_sampling) == (0, False)
    # Off: the tuner is used below 24 MHz too (an RTL-SDR Blog V4).
    assert rtl.set_frequency(3.5e6)
    assert rtl._device.direct == 0
    assert rtl.set_direct_sampling("bogus") is False
    assert rtl.direct_sampling_branch == DIRECT_SAMPLING_OFF


def test_changing_the_branch_above_24_mhz_waits_for_hf(rtl):
    rtl._device.calls.clear()
    assert rtl.set_direct_sampling("i")
    assert not [c for c in rtl._device.calls if c[0] == "direct"]
    assert rtl.set_frequency(10e6)
    assert rtl._device.direct == 1


def test_branch_chosen_before_opening_is_kept(rtl):
    dev = RTLSDRDevice()
    assert dev.set_direct_sampling("off")
    assert dev.open(0)
    assert dev.set_frequency(7e6)
    assert dev._device.direct == 0
    dev.close()


# --------------------------------------------------------------------------- #
# RTL-SDR bias tee and capabilities
# --------------------------------------------------------------------------- #
def test_bias_tee_is_switched_off_when_the_device_closes(rtl):
    handle = rtl._device
    assert rtl.set_bias_tee(True)
    assert handle.bias is True
    assert rtl.state.bias_tee_enabled
    rtl.close()
    assert handle.bias is False
    assert handle.closed


def test_rtl_capabilities(rtl):
    for cap in (
        DeviceCapability.AGC,
        DeviceCapability.FREQ_CORRECTION,
        DeviceCapability.BIAS_TEE,
        DeviceCapability.DIRECT_SAMPLE,
    ):
        assert rtl.has_capability(cap), cap
    listed = RTLSDRDevice.list_devices()
    assert DeviceCapability.AGC in listed[0].capabilities


# --------------------------------------------------------------------------- #
# HackRF One
# --------------------------------------------------------------------------- #
class FakeHackRF:
    """python_hackrf's PyHackrfDevice (the methods the driver uses)."""

    def __init__(self):
        self.calls = []
        self.antenna = False
        self.lna = self.vga = None
        self.amp = False

    def pyhackrf_serialno_read(self):
        return "0000000000beef"

    def pyhackrf_set_freq(self, freq):
        self.calls.append(("freq", freq))

    def pyhackrf_set_sample_rate(self, rate):
        self.calls.append(("rate", rate))

    def pyhackrf_set_baseband_filter_bandwidth(self, bw):
        self.calls.append(("bw", bw))

    def pyhackrf_set_lna_gain(self, gain):
        self.lna = gain

    def pyhackrf_set_vga_gain(self, gain):
        self.vga = gain

    def pyhackrf_set_amp_enable(self, on):
        self.amp = bool(on)

    def pyhackrf_set_antenna_enable(self, on):
        self.calls.append(("antenna", bool(on)))
        self.antenna = bool(on)

    def set_rx_callback(self, fn):
        self.calls.append(("rx_callback",))

    def pyhackrf_start_rx(self):
        self.calls.append(("start_rx",))

    def pyhackrf_stop_rx(self):
        self.calls.append(("stop_rx",))
        self.antenna = False  # the firmware: off when it goes idle

    def pyhackrf_close(self):
        self.calls.append(("close",))


@pytest.fixture
def hackrf(monkeypatch):
    fake = FakeHackRF()
    module = types.ModuleType("python_hackrf")
    backend = types.ModuleType("python_hackrf.pyhackrf")
    backend.pyhackrf_init = lambda: None
    backend.pyhackrf_open = lambda: fake
    backend.pyhackrf_compute_baseband_filter_bw = lambda bw: int(bw)
    module.pyhackrf = backend
    monkeypatch.setitem(sys.modules, "python_hackrf", module)
    monkeypatch.setitem(sys.modules, "python_hackrf.pyhackrf", backend)
    monkeypatch.setattr(hackrf_mod, "_initialized", False)
    dev = HackRFDevice()
    assert dev.open(0)
    yield dev, fake
    dev.close()


def test_hackrf_refuses_automatic_gain(hackrf):
    dev, fake = hackrf
    assert dev.set_gain(36)
    assert dev.set_gain_mode(True) is False
    assert dev.state.gain_mode == "manual"
    assert (dev.lna_gain, dev.vga_gain) == (32, 4)
    assert dev.set_gain_mode(False) is True
    assert not dev.has_capability(DeviceCapability.AGC)


def test_hackrf_lna_and_vga_read_back(hackrf):
    dev, fake = hackrf
    assert dev.set_lna_gain(24) and dev.set_vga_gain(31)
    assert (dev.lna_gain, dev.vga_gain) == (24, 30)
    assert (fake.lna, fake.vga) == (24, 30)
    assert dev.state.gain == 54


def test_hackrf_antenna_power_comes_back_on_each_reception(hackrf):
    dev, fake = hackrf
    assert dev.has_capability(DeviceCapability.BIAS_TEE)
    assert dev.set_bias_tee(True)
    assert fake.antenna and dev.state.bias_tee_enabled
    assert dev.start_rx(lambda samples: None)
    assert dev.stop_rx()
    assert fake.antenna is False  # the firmware switched it off
    fake.calls.clear()
    assert dev.start_rx(lambda samples: None)
    assert fake.antenna is True
    names = [c[0] for c in fake.calls]
    assert names.index("antenna") < names.index("start_rx")
    dev.stop_rx()


def test_hackrf_antenna_power_off_stays_off(hackrf):
    dev, fake = hackrf
    assert dev.set_bias_tee(True) and dev.set_bias_tee(False)
    fake.calls.clear()
    assert dev.start_rx(lambda samples: None)
    assert ("antenna", True) not in fake.calls
    dev.stop_rx()


def test_hackrf_antenna_power_refused_while_transmitting(hackrf):
    dev, fake = hackrf
    dev._state.is_transmitting = True
    try:
        assert dev.set_bias_tee(True) is False
        assert not fake.antenna
        assert dev.set_bias_tee(False) is True  # off is always fine
    finally:
        dev._state.is_transmitting = False


def test_hackrf_antenna_power_off_at_close(hackrf):
    dev, fake = hackrf
    assert dev.set_bias_tee(True)
    dev.close()
    assert fake.antenna is False
    assert fake.calls[-1] == ("close",)


def test_hackrf_without_antenna_power_in_the_binding(hackrf):
    dev, fake = hackrf
    del FakeHackRF.pyhackrf_set_antenna_enable
    try:
        assert dev.set_bias_tee(True) is False
        assert not dev.state.bias_tee_enabled
    finally:
        FakeHackRF.pyhackrf_set_antenna_enable = _antenna_method


_antenna_method = FakeHackRF.pyhackrf_set_antenna_enable


def test_queued_samples_can_be_discarded(rtl):
    import numpy as np

    for _ in range(3):
        rtl._sample_queue.put(np.zeros(8, np.complex64))
    assert rtl.discard_queued_samples() == 3
    assert rtl.read_samples(8, timeout=0) is None
    assert rtl.discard_queued_samples() == 0

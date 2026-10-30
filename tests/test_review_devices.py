"""Regression tests for the code review of the device area (DEV-1 ... DEV-7).

No hardware: the drivers run on fakes of their libraries. For the HackRF
One, :class:`FirmwareHackRF` adds what its firmware does to the RF path when
the transceiver mode changes (greatscottgadgets/hackrf v2024.02.1:
firmware/common/rf_path.c, user_config.c, hackrf_usb/usb_api_transceiver.c):

- hackrf_stop_rx/stop_tx (and hackrf_close) put it in OFF mode, and
  ``rf_path_set_direction(OFF)`` switches the antenna port power off (the
  user bias-T option for OFF is CLEAR by default) and bypasses the RF amp
  (``rf_path_set_lna(rf_path, 0)``);
- RX and TX leave the antenna power as it is (their bias-T option is NOP)
  and power the amp only if it isn't bypassed; nothing un-bypasses it but
  ``set_amp_enable(1)``;
- going from idle (OFF) straight to TX switches nothing off.

For the RTL-SDR, :class:`R820TRtlSdr` adds what librtlsdr does to an R820T
tuner (osmocom and rtl-sdr-blog librtlsdr.c, tuner_r82xx.c):
``rtlsdr_set_direct_sampling(dev, 0)`` runs ``tuner->init`` (``r82xx_init``),
which puts the LNA and mixer back on automatic gain, the VGA and the IF
filter back to their defaults; nothing sets the gain chosen before again.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sdr_module.devices.hackrf as hackrf_mod  # noqa: E402
from tests.hackrf_fakes import ThreadedFakeHackRF, install_fake_backend  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_device_settings import FakeRtlSdr, _DeviceCase  # noqa: E402
from tests.test_gui_second_receiver import _SecondCase  # noqa: E402
from tests.test_gui_station_tx import _Patch, _TxCase  # noqa: E402

# --------------------------------------------------------------------------- #
# A HackRF One whose firmware changes the RF path with the transceiver mode
# --------------------------------------------------------------------------- #


class FirmwareHackRF(ThreadedFakeHackRF):
    """The threaded python_hackrf fake plus the firmware's RF path (see the
    module notes). ``starts`` lists ``(mode, amp active, antenna powered)``
    at each start of a reception or a transmission."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = "off"  # the firmware boots into OFF mode
        self.amp_bypass = True  # rf_path_init: the amp bypassed
        self.antenna_power = False
        self.starts = []

    @property
    def amp_active(self) -> bool:
        """Whether the RF amp amplifies right now."""
        return self.mode in ("rx", "tx") and not self.amp_bypass

    def _enter(self, mode: str) -> None:
        if self.mode in ("rx", "tx"):
            # transceiver_shutdown -> rf_path_set_direction(OFF)
            self.antenna_power = False
            self.amp_bypass = True
        self.mode = mode
        if mode in ("rx", "tx"):
            self.starts.append((mode, not self.amp_bypass, self.antenna_power))

    def pyhackrf_set_amp_enable(self, on):
        super().pyhackrf_set_amp_enable(on)
        self.amp_bypass = not on

    def pyhackrf_set_antenna_enable(self, on):
        super().pyhackrf_set_antenna_enable(on)
        self.antenna_power = bool(on)

    def pyhackrf_start_rx(self):
        super().pyhackrf_start_rx()
        self._enter("rx")

    def pyhackrf_stop_rx(self):
        super().pyhackrf_stop_rx()
        self._enter("off")

    def pyhackrf_start_tx(self):
        self._enter("tx")
        super().pyhackrf_start_tx()

    def pyhackrf_stop_tx(self):
        super().pyhackrf_stop_tx()
        self._enter("off")

    def pyhackrf_close(self):
        self._enter("off")
        super().pyhackrf_close()

    def tx_starts(self):
        return [start for start in self.starts if start[0] == "tx"]


def open_firmware_hackrf(patch, rate=2.4e6, **fake_args):
    """A HackRFDevice opened on a :class:`FirmwareHackRF` (as
    tests.hackrf_fakes.open_hackrf does). Returns (device, fake)."""
    from sdr_module.devices.hackrf import HackRFDevice

    fake = FirmwareHackRF(**fake_args)
    install_fake_backend(patch, fake)
    device = HackRFDevice()
    assert device.open(0)
    device.set_sample_rate(rate)
    device.set_bandwidth(0.75 * rate)
    return device, fake


@pytest.fixture
def fw_hackrf(monkeypatch):
    # (The lockouts and licence checks are tested on their own.)
    monkeypatch.setattr(hackrf_mod, "is_tx_allowed", lambda *a, **k: (True, None))
    device, fake = open_firmware_hackrf(monkeypatch)
    yield device, fake
    device.close()


def _transmit_once(device):
    """Send a few samples (start_tx, let the fake take them, stop_tx)."""
    blocks = [np.full(64, 0.5 + 0.5j, dtype=np.complex64)]
    assert device.start_tx(
        lambda: blocks.pop() if blocks else None, bandwidth_hz=150.0, mode="CW"
    )
    assert device.wait_tx_done(5.0)
    device.stop_tx()


# --------------------------------------------------------------------------- #
# DEV-1: the HackRF's RF amp is set again whenever a stream starts
# --------------------------------------------------------------------------- #


def test_dev1_the_amp_comes_back_after_every_stop(fw_hackrf):
    device, fake = fw_hackrf
    assert device.set_amp(True)
    for _ in range(3):  # Start, then Stop/Start twice
        assert device.start_rx(lambda samples: None)
        assert fake.amp_active
        assert device.stop_rx()
        assert not fake.amp_active  # (the firmware bypassed it: OFF mode)
    assert [amp for _mode, amp, _antenna in fake.starts] == [True, True, True]
    assert device.amp_enabled and device.state.amp_enabled


def test_dev1_receiving_after_a_transmission_has_the_amp(fw_hackrf):
    device, fake = fw_hackrf
    assert device.set_amp(True)
    _transmit_once(device)
    assert device.start_rx(lambda samples: None)
    assert fake.amp_active
    device.stop_rx()


def test_dev1_a_transmission_after_receiving_has_the_amp_set(fw_hackrf):
    """The amp is one for both directions: transmitting after a reception
    (library use: write_samples) uses it as it is set, not bypassed."""
    device, fake = fw_hackrf
    assert device.set_amp(True)
    assert device.start_rx(lambda samples: None)
    device.stop_rx()
    _transmit_once(device)
    assert fake.tx_starts() == [("tx", True, False)]


def test_dev1_an_amp_switched_off_stays_off(fw_hackrf):
    device, fake = fw_hackrf
    assert device.set_amp(True) and device.set_amp(False)
    assert device.start_rx(lambda samples: None)
    assert not fake.amp_active
    device.stop_rx()
    _transmit_once(device)
    assert [amp for _mode, amp, _antenna in fake.starts] == [False, False]


# --------------------------------------------------------------------------- #
# DEV-2: the HackRF's antenna port is never powered while transmitting
# --------------------------------------------------------------------------- #


def test_dev2_bias_tee_set_while_idle_is_off_on_the_air(fw_hackrf):
    device, fake = fw_hackrf
    assert device.set_bias_tee(True)  # while idle: powered at once
    assert fake.antenna_power
    _transmit_once(device)
    assert fake.tx_starts() == [("tx", False, False)]
    assert device.state.bias_tee_enabled  # (still on for receiving)
    assert device.start_rx(lambda samples: None)
    assert fake.antenna_power  # powered again to receive
    device.stop_rx()


def test_dev2_a_transmission_is_refused_if_the_power_wont_go_off(fw_hackrf):
    device, fake = fw_hackrf
    assert device.set_bias_tee(True)

    def broken(on):
        raise OSError("USB error")

    fake.pyhackrf_set_antenna_enable = broken
    assert device.start_tx(lambda: None, bandwidth_hz=150.0, mode="CW") is False
    assert "antenna" in (device.last_tx_error or "")
    assert not device.state.is_transmitting
    assert fake.tx_starts() == []


def test_dev2_without_the_bias_tee_a_failed_switch_off_doesnt_matter(fw_hackrf):
    device, fake = fw_hackrf

    def broken(on):
        raise OSError("USB error")

    fake.pyhackrf_set_antenna_enable = broken
    _transmit_once(device)
    assert len(fake.tx_starts()) == 1


# --------------------------------------------------------------------------- #
# DEV-1 / DEV-2 in the window: Receiver group, Stop/Start, rate, Ham ID
# --------------------------------------------------------------------------- #


class _FirmwareCase(_TxCase):
    """The transmit tests' window with a HackRF on :class:`FirmwareHackRF`,
    connected as Device > Connect... does."""

    def connect_fw(self, rate=10e6, freq=146.52e6):
        if self.win._current_frequency() != freq:
            self.win.set_frequency(freq)
        dev, fake = open_firmware_hackrf(self.patch, rate=rate)
        self.win._device = dev
        self.win._demo_mode = False
        self.note = self.win._apply_controls_to_device()
        self.win._refresh_state_ui()
        self.addCleanup(self._release)
        return dev, fake


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDev1HackRFAmpInTheWindow(_FirmwareCase):
    def test_the_receive_amp_stays_on_through_stops_rate_changes_and_ids(self):
        self.panel.set_callsign("W1AW")
        dev, fake = self.connect_fw()
        receiver = self.win._control_panel
        receiver._amp_check.setChecked(True)  # Receiver group: RF amp
        self.panel._amp_check.setChecked(True)  # Ham ID > TX Power: amp too
        active = []
        self.win._start_acquisition()
        active.append(fake.amp_active)
        self.win._stop_acquisition()  # Space twice
        self.win._start_acquisition()
        active.append(fake.amp_active)
        self.assertTrue(self.win.set_sample_rate(8e6))  # live rate change
        active.append(fake.amp_active)
        self.assertTrue(self.station.send_station_id())
        self.wait_done(30)
        self.assertTrue(self.win._is_running)  # (receiving again)
        active.append(fake.amp_active)
        self.win._stop_acquisition()
        self.assertEqual(active, [True, True, True, True])
        # What the window says is what the HackRF does.
        self.assertTrue(receiver.hackrf_gains()[2] and dev.amp_enabled)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDev2HackRFBiasTeeOnTheAir(_FirmwareCase):
    def test_a_remembered_bias_tee_is_off_for_an_id_sent_before_receiving(self):
        self.store.values["device/hackrf/bias_tee"] = True
        self.panel.set_callsign("W1AW")
        dev, fake = self.connect_fw()
        self.assertIn("The bias tee is on", self.note)
        self.assertTrue(fake.antenna_power)
        self.assertTrue(self.station.send_station_id())
        self.wait_done(30)
        self.assertEqual([antenna for _m, _a, antenna in fake.tx_starts()], [False])
        self.win._start_acquisition()  # receiving powers it again
        self.assertTrue(fake.antenna_power)
        self.assertTrue(self.win._control_panel._bias_check.isChecked())
        self.win._stop_acquisition()

    def test_ticked_while_stopped_then_an_id(self):
        self.panel.set_callsign("W1AW")
        dev, fake = self.connect_fw()
        self.win._control_panel._bias_check.setChecked(True)
        self.assertTrue(dev.state.bias_tee_enabled and fake.antenna_power)
        self.assertTrue(self.station.send_station_id())
        self.wait_done(30)
        self.assertEqual([antenna for _m, _a, antenna in fake.tx_starts()], [False])


# --------------------------------------------------------------------------- #
# DEV-3: the RTL-SDR's tuner gets its gain and IF filter back after HF
# --------------------------------------------------------------------------- #


class R820TRtlSdr(FakeRtlSdr):
    """The tests' pyrtlsdr fake plus an R820T's gain and IF filter, as
    librtlsdr leaves them (see the module notes)."""

    def __init__(self, device_index=0):
        # rtlsdr_open -> tuner->init: automatic gain, defaults.
        self.tuner_agc = True
        self.vga = "init"  # "init", "agc" (fixed for AGC) or "manual"
        self.manual_gain = 0.0  # librtlsdr's dev->gain
        self.if_filter: object = "init"  # "init", "rate" or a width (Hz)
        self.tuner_bw = 0  # librtlsdr's dev->bw (0: follows the rate)
        self.rate = 0.0
        super().__init__(device_index)

    # pyrtlsdr's properties
    @property
    def gain(self):
        return self.manual_gain

    @gain.setter
    def gain(self, value):
        if isinstance(value, str) and value == "auto":
            self.tuner_agc, self.vga = True, "agc"
        else:
            self.tuner_agc, self.vga = False, "manual"
            self.manual_gain = float(value)

    @property
    def sample_rate(self):
        return self.rate

    @sample_rate.setter
    def sample_rate(self, rate):
        self.rate = float(rate)
        self.if_filter = self.tuner_bw or "rate"  # (rtlsdr_set_sample_rate)

    def set_bandwidth(self, bw):
        self.tuner_bw = int(bw)
        self.if_filter = self.tuner_bw or "rate"

    def set_direct_sampling(self, mode):
        super().set_direct_sampling(mode)
        if not mode:  # tuner->init (r82xx_init)
            self.tuner_agc, self.vga, self.if_filter = True, "init", "init"

    def tuner(self) -> str:
        """What the tuner's gain does: "manual 29.7 dB", "auto" (as
        ``gain = "auto"`` sets it), or "auto after init"."""
        if not self.tuner_agc:
            return f"manual {self.manual_gain:g} dB"
        return "auto" if self.vga == "agc" else "auto after init"


@pytest.fixture
def r820t(monkeypatch):
    from sdr_module.devices.rtlsdr import RTLSDRDevice

    rtlsdr = types.ModuleType("rtlsdr")
    librtlsdr = types.ModuleType("rtlsdr.librtlsdr")
    librtlsdr.rtlsdr_get_device_count = lambda: 1
    rtlsdr.RtlSdr = R820TRtlSdr
    rtlsdr.librtlsdr = librtlsdr
    monkeypatch.setitem(sys.modules, "rtlsdr", rtlsdr)
    monkeypatch.setitem(sys.modules, "rtlsdr.librtlsdr", librtlsdr)
    monkeypatch.setattr(FakeRtlSdr, "gains_db", None)
    monkeypatch.setattr(FakeRtlSdr, "fail", set())
    device = RTLSDRDevice()
    assert device.open(0)
    yield device, device._device
    device.close()


def _visit_hf(device, tuner):
    assert device.set_frequency(7.1e6)  # 40 m: direct sampling (Q branch)
    assert tuner.direct == 2
    assert device.set_frequency(145.5e6)  # the tuner again
    assert tuner.direct == 0


def test_dev3_the_manual_gain_is_set_again_after_hf(r820t):
    device, tuner = r820t
    assert device.set_gain_mode(False) and device.set_gain(29.7)
    assert tuner.tuner() == "manual 29.7 dB"
    _visit_hf(device, tuner)
    assert tuner.tuner() == "manual 29.7 dB"
    assert (device.state.gain_mode, device.state.gain) == ("manual", 29.7)


def test_dev3_automatic_gain_is_set_up_again_too(r820t):
    device, tuner = r820t
    assert device.set_gain_mode(True)
    _visit_hf(device, tuner)
    assert tuner.tuner() == "auto"  # (not the VGA r82xx_init leaves)


def test_dev3_the_if_filter_is_set_again(r820t):
    device, tuner = r820t
    assert device.set_bandwidth(1.5e6)
    _visit_hf(device, tuner)
    assert tuner.if_filter == 1_500_000
    assert device.set_bandwidth(0)  # automatic: it follows the sample rate
    _visit_hf(device, tuner)
    assert tuner.if_filter == "rate"


def test_dev3_direct_sampling_switched_off_at_hf_sets_the_gain_too(r820t):
    """Choosing "Off (V4 or upconverter)" while tuned to HF leaves direct
    sampling there and then."""
    device, tuner = r820t
    assert device.set_gain_mode(False) and device.set_gain(29.7)
    assert device.set_frequency(7.1e6)
    assert device.set_direct_sampling("off")
    assert tuner.direct == 0
    assert tuner.tuner() == "manual 29.7 dB"


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDev3RtlGainAfterHfInTheWindow(_DeviceCase):
    def test_the_gain_shown_is_the_gain_applied_after_a_visit_to_hf(self):
        self.patch.setattr(sys.modules["rtlsdr"], "RtlSdr", R820TRtlSdr)
        dev = self.rtl()
        tuner = dev._device
        self.panel.set_agc_enabled(False)
        self.win.set_gain(29.7)
        self.assertTrue(self.win.set_frequency(145.5e6))
        self.assertEqual(tuner.tuner(), "manual 29.7 dB")
        self.assertTrue(self.win.set_frequency(7.1e6))
        self.assertEqual(tuner.direct, 2)
        self.assertTrue(self.win.set_frequency(145.5e6))
        self.assertEqual(tuner.direct, 0)
        self.assertEqual(tuner.tuner(), "manual 29.7 dB")
        self.assertEqual(self.panel._gain_label.text(), "29.7 dB")
        self.assertFalse(self.panel.is_agc_enabled())
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("gain"), "29.7 dB")


# --------------------------------------------------------------------------- #
# The second receiver on the real drivers (DEV-4, DEV-6, DEV-7)
# --------------------------------------------------------------------------- #


def _install_rtl(patch, tuner_class=FakeRtlSdr):
    """A fake ``rtlsdr`` module (pyrtlsdr's surface) opening ``tuner_class``."""
    patch.setattr(FakeRtlSdr, "gains_db", None)
    patch.setattr(FakeRtlSdr, "fail", set())
    rtlsdr = types.ModuleType("rtlsdr")
    librtlsdr = types.ModuleType("rtlsdr.librtlsdr")
    librtlsdr.rtlsdr_get_device_count = lambda: 1
    rtlsdr.RtlSdr = tuner_class
    rtlsdr.librtlsdr = librtlsdr
    patch.setitem(sys.modules, "rtlsdr", rtlsdr)
    patch.setitem(sys.modules, "rtlsdr.librtlsdr", librtlsdr)


class _SecondDriverCase(_SecondCase):
    """The second receiver with a real RTLSDRDevice or HackRFDevice on the
    fakes of their libraries."""

    def setUp(self):
        super().setUp()
        self.patch = _Patch()
        self.addCleanup(self.patch.undo)

    def use_hackrf(self):
        from sdr_module.devices.hackrf import HackRFDevice

        fake = FirmwareHackRF()
        install_fake_backend(self.patch, fake)
        self.win._hardware_classes = [HackRFDevice]
        return fake

    def use_rtl(self, tuner_class=FakeRtlSdr):
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        _install_rtl(self.patch, tuner_class)
        self.win._hardware_classes = [RTLSDRDevice]

    def choose(self, kind):
        rx = self.rx
        rx._refresh_sources()
        index = next(i for i, s in enumerate(rx._sources) if s.kind == kind)
        rx._source_combo.setCurrentIndex(index)
        rx._on_source_activated(index)

    def start_on(self, kind):
        self.choose(kind)
        self.assertTrue(self.rx.start(), self.rx.message())
        return self.rx.device

    def shown(self, widget):
        return self.rx._device_form.isRowVisible(widget)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDev4SecondReceiverGain(_SecondDriverCase):
    def test_a_hackrf_gets_lna_vga_and_the_rf_amp(self):
        fake = self.use_hackrf()
        rx = self.rx
        self.choose("hackrf")
        # The HackRF's own controls, before it is even open.
        self.assertFalse(self.shown(rx._gain_slider.parentWidget()))
        self.assertTrue(self.shown(rx._lna_slider.parentWidget()))
        self.assertTrue(self.shown(rx._vga_slider.parentWidget()))
        self.assertTrue(self.shown(rx._amp_check))
        dev = self.start_on("hackrf")
        self.assertEqual((fake.lna, fake.vga, fake.amp), (16, 20, False))
        rx._lna_slider.setValue(rx._lna_slider.maximum())  # 40 dB
        rx._vga_slider.setValue(rx._vga_slider.maximum())  # 62 dB
        self.assertEqual((dev.lna_gain, dev.vga_gain), (40, 62))
        self.assertEqual((fake.lna, fake.vga), (40, 62))
        self.assertEqual(rx._lna_label.text(), "40 dB")
        self.assertEqual(rx._vga_label.text(), "62 dB")
        rx._lna_slider.setValue(4)  # the usual LNA 32 + VGA 30
        rx._vga_slider.setValue(15)
        self.assertEqual((fake.lna, fake.vga), (32, 30))
        self.assertFalse(self.shown(rx._amp_warning))
        rx._amp_check.setChecked(True)
        self.assertTrue(fake.amp and dev.amp_enabled and fake.amp_active)
        self.assertTrue(self.shown(rx._amp_warning))
        rx._amp_check.setChecked(False)
        self.assertFalse(fake.amp or dev.amp_enabled)

    def test_the_hackrf_stages_are_remembered(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        fake = self.use_hackrf()
        self.start_on("hackrf")
        self.rx._lna_slider.setValue(3)  # 24 dB
        self.rx._vga_slider.setValue(17)  # 34 dB
        self.rx._amp_check.setChecked(True)
        self.rx.shutdown()
        self.assertTrue(fake.closed)
        values = self.store.values
        self.assertEqual(values["device/second_receiver/hackrf_lna"], 24)
        self.assertEqual(values["device/second_receiver/hackrf_vga"], 34)
        self.assertTrue(values["device/second_receiver/hackrf_amp"])
        fake.closed = False
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertTrue(again.start(), again.message())
        self.assertEqual((fake.lna, fake.vga, fake.amp), (24, 34, True))
        again.stop()

    def test_damaged_hackrf_stages_fall_back_to_the_defaults(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.rx.shutdown()
        self.store.values.update(
            {
                "device/second_receiver/hackrf_lna": float("nan"),
                "device/second_receiver/hackrf_vga": 1000.0,
                "device/second_receiver/gain_db": float("inf"),
            }
        )
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._hackrf_gains(), (16, 62, False))
        self.assertEqual(again._gain_slider.value(), 20)

    def test_an_rtl_sdr_steps_through_its_tuners_gains_and_shows_them(self):
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        self.use_rtl()
        rx = self.rx
        dev = self.start_on("rtlsdr")
        slider = rx._gain_slider
        table = dev.get_tuner_gains()
        self.assertEqual(table, RTLSDRDevice.VALID_GAINS)
        self.assertEqual((slider.minimum(), slider.maximum()), (0, len(table) - 1))
        self.assertFalse(self.shown(rx._lna_slider.parentWidget()))
        for index in (16, 26, 27, len(table) - 1):  # 29.7, 44.5, 48, 49.6 dB
            slider.setValue(index)
            self.assertEqual(dev.state.gain, table[index])
            self.assertEqual(dev._device.gain, table[index])
            expected = f"{table[index]:.1f}".rstrip("0").rstrip(".") + " dB"
            self.assertEqual(rx._gain_label.text(), expected)
        self.assertEqual(self.store.values["device/second_receiver/gain_db"], 49.6)

    def test_the_gain_carries_over_between_devices_to_the_nearest_step(self):
        self.use_rtl()
        rx = self.rx
        rx._gain_slider.setValue(30)  # the demo device: 30 dB
        self.start_on("rtlsdr")
        self.assertEqual(rx.device.state.gain, 29.7)
        self.assertEqual(rx._gain_label.text(), "29.7 dB")
        rx.stop()
        self.choose("demo")
        self.assertEqual(rx._gain_slider.value(), 30)  # 29.7 -> 30 dB
        self.assertEqual(rx._gain_label.text(), "30 dB")


if __name__ == "__main__":
    unittest.main()

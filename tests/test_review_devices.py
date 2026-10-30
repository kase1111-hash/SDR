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
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sdr_module.devices.hackrf as hackrf_mod  # noqa: E402
from tests.hackrf_fakes import ThreadedFakeHackRF, install_fake_backend  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()

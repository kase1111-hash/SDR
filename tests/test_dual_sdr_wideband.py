"""DualSDRController's wideband scan mode (K-X1).

Before, ``set_mode(OperationMode.WIDEBAND_SCAN)`` raised
``NotImplementedError``: the controller couldn't have its two devices sweep
a range together. Now it can (``start_wideband_scan``), with progress, the
signals merged in frequency order, and stop. Relay stays deliberately
unimplemented, and its message says why in plain words.
"""

import logging

import pytest

from sdr_module.core.dual_sdr import (
    NOT_IMPLEMENTED_MODES,
    DualSDRController,
    OperationMode,
)
from sdr_module.core.wideband_scan import CANCELLED, COMPLETE
from tests.test_dual_sdr import MockHackRFDevice, MockRTLSDRDevice
from tests.tuner_fakes import STATIONS, FakeTuner

FAST = dict(settle_s=0.0, read_timeout_s=0.5)


def _controller(first=None, second=None):
    controller = DualSDRController()
    assert controller.use_devices(
        first or FakeTuner("RTL-SDR", low_hz=24e6, high_hz=1766e6),
        second or FakeTuner("HackRF One", low_hz=1e6, high_hz=6e9),
    )
    return controller


def _freqs(hits):
    return [round(h.frequency_hz) for h in hits]


class TestMode:
    def test_wideband_scan_is_implemented(self):
        assert OperationMode.WIDEBAND_SCAN.implemented
        assert OperationMode.WIDEBAND_SCAN in DualSDRController.supported_modes()
        assert OperationMode.WIDEBAND_SCAN not in NOT_IMPLEMENTED_MODES

    def test_set_mode_needs_a_device(self):
        assert not DualSDRController().set_mode(OperationMode.WIDEBAND_SCAN)
        controller = DualSDRController()
        controller._rtlsdr = MockRTLSDRDevice()
        controller._hackrf = MockHackRFDevice()
        assert controller.set_mode(OperationMode.WIDEBAND_SCAN)
        assert controller.state.mode == OperationMode.WIDEBAND_SCAN

    def test_relay_says_plainly_why_it_is_not_there(self):
        controller = _controller()
        with pytest.raises(NotImplementedError) as raised:
            controller.set_mode(OperationMode.RELAY)
        message = str(raised.value)
        assert "deliberately not implemented" in message
        assert "legal" in message and "97.113" in message
        assert "Transmit Text" in message
        assert controller.state.mode == OperationMode.DUAL_RX


class TestScan:
    def test_two_devices_sweep_the_range_together(self):
        controller = _controller()
        ended = []
        assert controller.start_wideband_scan(
            88e6, 108e6, 100e3, on_finished=ended.append, **FAST
        )
        scan = controller.wideband_scan
        assert controller.state.mode == OperationMode.WIDEBAND_SCAN
        assert scan.wait(20.0)
        assert ended == [COMPLETE] and scan.reason == COMPLETE
        assert _freqs(scan.hits) == sorted(STATIONS)
        assert controller.rtlsdr.tuned and controller.hackrf.tuned
        state = controller.state
        assert not state.rtlsdr_streaming and not state.hackrf_streaming
        # Where the scan left them.
        assert state.rtlsdr_frequency == controller.rtlsdr.tuned[-1]
        assert state.hackrf_frequency == controller.hackrf.tuned[-1]
        status = controller.get_status()["wideband_scan"]
        assert status == {
            "running": False,
            "pass": 1,
            "progress": pytest.approx(1.0),
            "found": len(STATIONS),
            "reason": COMPLETE,
        }

    def test_the_devices_are_busy_while_it_scans(self, caplog):
        slow = FakeTuner("Slow", deliver=False)
        controller = _controller(slow)
        assert controller.start_wideband_scan(88e6, 108e6, 100e3, read_timeout_s=60)
        assert slow.reading.wait(5.0)
        state = controller.state
        assert state.rtlsdr_streaming and state.hackrf_streaming
        with caplog.at_level(logging.ERROR):
            assert not controller.start_dual_rx()
        assert "wideband scan" in caplog.text
        assert not controller.start_full_duplex()
        assert not controller.set_mode(OperationMode.DUAL_RX)
        assert not controller.use_devices(FakeTuner(), FakeTuner())
        # The scan tunes them: not retuned behind its back.
        assert not controller.set_rtlsdr_frequency(433.92e6)
        assert not controller.set_hackrf_frequency(433.92e6)
        assert 433.92e6 not in slow.tuned
        assert not controller.start_wideband_scan(88e6, 108e6, 100e3)
        assert controller.stop_wideband_scan(10.0)
        assert controller.wideband_scan.reason == CANCELLED
        assert slow.stopped == 1
        assert not controller.state.rtlsdr_streaming

    def test_stop_all_stops_the_scan(self):
        slow = FakeTuner("Slow", deliver=False)
        controller = _controller(slow)
        assert controller.start_wideband_scan(88e6, 108e6, 100e3, read_timeout_s=60)
        assert slow.reading.wait(5.0)
        controller.stop_all()
        scan = controller.wideband_scan
        assert scan.wait(0) and scan.reason == CANCELLED
        assert (slow.stopped, controller.hackrf.stopped) == (1, 1)
        state = controller.state
        assert not state.rtlsdr_streaming and not state.hackrf_streaming
        # Then dual receive works again.
        assert controller.start_dual_rx()
        controller.stop_all()

    def test_not_while_receiving(self, caplog):
        controller = _controller()
        assert controller.start_dual_rx()
        with caplog.at_level(logging.WARNING):
            assert not controller.start_wideband_scan(88e6, 108e6, 100e3)
        assert "Stop streaming" in caplog.text
        controller.stop_all()
        assert controller.start_wideband_scan(88e6, 108e6, 100e3, **FAST)
        assert controller.wideband_scan.wait(20.0)

    def test_refusals_say_why(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert not DualSDRController().start_wideband_scan(88e6, 108e6, 1e5)
        assert "needs a device" in caplog.text
        caplog.clear()
        controller = _controller()
        with caplog.at_level(logging.ERROR):
            assert not controller.start_wideband_scan(108e6, 88e6, 100e3)
        assert "above its start" in caplog.text
        assert controller.wideband_scan is None
        assert not controller.state.rtlsdr_streaming

    def test_a_scan_that_cannot_start_changes_nothing(self, monkeypatch, caplog):
        from sdr_module.core import wideband_scan

        def no_thread(self):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(wideband_scan.WidebandScan, "start", no_thread)
        controller = _controller()
        with caplog.at_level(logging.ERROR):
            assert not controller.start_wideband_scan(88e6, 108e6, 100e3)
        assert "can't start new thread" in caplog.text
        assert controller.wideband_scan is None
        state = controller.state
        assert not state.rtlsdr_streaming and not state.hackrf_streaming
        assert controller.start_dual_rx()
        controller.stop_all()

    def test_continuous_until_stopped_from_a_callback(self):
        controller = _controller()
        passes = []

        def on_pass(report):
            passes.append(report)
            if report.number == 2:
                assert not controller.stop_wideband_scan()  # (its own thread)

        assert controller.start_wideband_scan(
            88e6, 108e6, 100e3, continuous=True, on_pass=on_pass, **FAST
        )
        assert controller.wideband_scan.wait(30.0)
        assert [p.number for p in passes] == [1, 2]
        assert controller.wideband_scan.reason == CANCELLED

    def test_one_device_scans_alone(self):
        controller = DualSDRController()
        controller.use_devices(None, FakeTuner("HackRF One"))
        assert controller.start_wideband_scan(88e6, 108e6, 100e3, **FAST)
        assert controller.wideband_scan.wait(20.0)
        assert _freqs(controller.wideband_scan.hits) == sorted(STATIONS)

    def test_two_demo_devices(self, monkeypatch):
        """As the README shows it: two demo devices through PolledDevice."""
        from sdr_module.devices.polled import PolledDevice
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)
        controller = DualSDRController()
        controller.use_devices(
            PolledDevice(MockDevice(), "Demo A"), PolledDevice(MockDevice(), "Demo B")
        )
        assert controller.start_wideband_scan(99e6, 101e6, 100e3, settle_s=0.03)
        assert controller.wideband_scan.wait(20.0)
        assert 100.1e6 in _freqs(controller.wideband_scan.hits)
        assert not controller.rtlsdr.state.is_streaming
        assert not controller.hackrf.state.is_streaming

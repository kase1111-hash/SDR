"""``sdr-scan scan`` without ``--input``: a live sweep of a connected SDR.

The device is mocked: a receiver that synthesises carriers at known
frequencies around wherever it is tuned, with a little noise, behind a
fake DeviceManager. One test drives the real RTL-SDR driver through a fake
``rtlsdr`` (pyrtlsdr) module, counting how often it is opened and read.
"""

import sys
import types
from dataclasses import dataclass, field
from typing import List
from unittest import mock

import numpy as np
import pytest

from sdr_module import cli
from sdr_module.cli import main

STATIONS = {88.5e6: -30.0, 93.3e6: -25.0, 100.1e6: -20.0, 107.9e6: -40.0}


class FakeReceiver:
    """Synthesises ``STATIONS`` around its centre frequency."""

    def __init__(self, stations=None, spec=None, deliver=True):
        self.stations = dict(STATIONS if stations is None else stations)
        self.spec = spec
        self.deliver = deliver
        self.sample_rate = 0.0
        self.frequency = 0.0
        self.tuned: List[float] = []
        self.started = self.stopped = 0
        self.gain_mode = None
        self.gain = None
        self.reads = 0
        self._rng = np.random.default_rng(3)
        self._queued = False

    def set_sample_rate(self, rate):
        self.sample_rate = float(rate)
        return True

    def set_gain_mode(self, auto):
        self.gain_mode = bool(auto)
        return True

    def set_gain(self, gain):
        self.gain = float(gain)
        return True

    def set_frequency(self, freq):
        if self.spec is not None and not (
            self.spec.freq_min <= freq <= self.spec.freq_max
        ):
            return False
        self.frequency = float(freq)
        self.tuned.append(float(freq))
        self._queued = True  # a stale block, as a driver would have queued
        return True

    def start_rx(self):
        self.started += 1
        return True

    def stop_rx(self):
        self.stopped += 1
        return True

    def read_samples(self, n, timeout=1.0):
        self.reads += 1
        if not self.deliver:
            return None
        if timeout == 0.0 and not self._queued:
            return None  # nothing queued: a drain ends here
        self._queued = False
        count = 32768
        t = np.arange(count) / self.sample_rate
        x = (
            self._rng.standard_normal(count) + 1j * self._rng.standard_normal(count)
        ) * 1e-4
        for freq, level in self.stations.items():
            offset = freq - self.frequency
            if abs(offset) < 0.5 * self.sample_rate:
                x = x + 10 ** (level / 20) * np.exp(2j * np.pi * offset * t)
        return x.astype(np.complex64)


@dataclass
class _Info:
    name: str = "Fake RTL-SDR"
    index: int = 0


@dataclass
class _Detected:
    info: _Info
    device_class: type
    is_available: bool = True


@dataclass
class FakeManager:
    device: object = None
    found: list = field(default_factory=list)
    closed: int = 0

    DEVICE_TYPES = {"rtlsdr": FakeReceiver}

    def scan_devices(self):
        return list(self.found)

    def open_device(self, device_type, index=0, config=None):
        assert device_type == "rtlsdr"
        return self.device

    def close_all(self):
        self.closed += 1


@pytest.fixture
def no_settle(monkeypatch):
    monkeypatch.setattr(cli, "_LIVE_SETTLE_S", 0.0)


def _patch_manager(monkeypatch, manager):
    from sdr_module.core import device_manager

    monkeypatch.setattr(device_manager, "DeviceManager", lambda: manager)


def test_without_a_device_it_fails_and_says_why(monkeypatch, capsys):
    _patch_manager(monkeypatch, FakeManager())
    rc = main(["scan", "--start", "88", "--end", "108", "--single"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No SDR device is connected" in out
    assert "--input" in out  # what to do instead


def test_single_sweep_finds_every_station(monkeypatch, capsys, no_settle):
    dev = FakeReceiver()
    manager = FakeManager(dev, [_Detected(_Info(), FakeReceiver)])
    _patch_manager(monkeypatch, manager)
    rc = main(["scan", "--start", "88", "--end", "108", "--step", "100", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    for freq in STATIONS:
        assert f"{freq / 1e6:10.4f} MHz" in out, out
    assert "Sweep 1 done" in out
    assert "4 signal(s)" in out
    # It stepped by about 0.8x the rate and covered the range.
    steps = np.diff(dev.tuned)
    assert np.all(steps > 0) and np.all(steps <= 0.8 * dev.sample_rate + 1)
    assert dev.tuned[0] - 0.4 * dev.sample_rate <= 88e6
    assert dev.tuned[-1] + 0.4 * dev.sample_rate >= 108e6
    # Started, stopped and released; automatic gain by default.
    assert (dev.started, dev.stopped, manager.closed) == (1, 1, 1)
    assert dev.gain_mode is True
    assert "FM Broadcast" in out  # the band of each signal


def test_threshold_and_gain_options(monkeypatch, capsys, no_settle):
    dev = FakeReceiver()
    _patch_manager(monkeypatch, FakeManager(dev, [_Detected(_Info(), FakeReceiver)]))
    rc = main(
        [
            "scan",
            "--start",
            "88",
            "--end",
            "108",
            "--single",
            "--threshold",
            "-28",
            "--gain",
            "30",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Threshold: -28.0 dBFS" in out
    # Only the stations above -28 dBFS.
    assert "93.3000 MHz" in out and "100.1000 MHz" in out
    assert "88.5000 MHz" not in out and "107.9000 MHz" not in out
    assert (dev.gain_mode, dev.gain) == (False, 30.0)


def test_continuous_sweeps_until_ctrl_c(monkeypatch, capsys, no_settle):
    dev = FakeReceiver()
    manager = FakeManager(dev, [_Detected(_Info(), FakeReceiver)])
    _patch_manager(monkeypatch, manager)
    real_set = dev.set_frequency
    calls = []

    def set_frequency(freq):
        calls.append(freq)
        if len(calls) > 30:  # into the third sweep
            raise KeyboardInterrupt
        return real_set(freq)

    dev.set_frequency = set_frequency
    rc = main(["scan", "--start", "88", "--end", "108", "--step", "100"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Sweep 1 done" in out and "Sweep 2 done" in out
    assert "Stopped." in out
    # Each station is listed once, however many sweeps saw it.
    assert out.count("100.1000 MHz") == 1
    assert (dev.stopped, manager.closed) == (1, 1)


def test_range_is_kept_inside_the_device(monkeypatch, capsys, no_settle):
    spec = types.SimpleNamespace(freq_min=24e6, freq_max=1766e6)
    dev = FakeReceiver(stations={1765e6: -30.0}, spec=spec)
    _patch_manager(monkeypatch, FakeManager(dev, [_Detected(_Info(), FakeReceiver)]))
    rc = main(["scan", "--start", "1760", "--end", "1800", "--step", "100", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "limited to Fake RTL-SDR's range" in out
    assert "1765.0000 MHz" in out
    assert max(dev.tuned) <= 1766e6

    dev = FakeReceiver(spec=spec)
    _patch_manager(monkeypatch, FakeManager(dev, [_Detected(_Info(), FakeReceiver)]))
    rc = main(["scan", "--start", "1800", "--end", "1900", "--single"])
    assert rc == 1
    assert "tunes 24.000 - 1766.000 MHz" in capsys.readouterr().out


def test_a_device_that_delivers_nothing_is_reported(monkeypatch, capsys, no_settle):
    monkeypatch.setattr(cli, "_LIVE_READ_TIMEOUT_S", 0.01)
    dev = FakeReceiver(deliver=False)
    manager = FakeManager(dev, [_Detected(_Info(), FakeReceiver)])
    _patch_manager(monkeypatch, manager)
    rc = main(["scan", "--start", "100", "--end", "101", "--single"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "returned no samples" in out
    assert (dev.stopped, manager.closed) == (1, 1)


def test_bad_range_and_threshold_are_refused(monkeypatch, capsys):
    _patch_manager(monkeypatch, FakeManager())
    assert main(["scan", "--start", "108", "--end", "88", "--single"]) == 1
    assert "--end must be above --start" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["scan", "--threshold", "loud"])


def test_offline_scan_accepts_auto_threshold(tmp_path, capsys):
    from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

    fs = 2.4e6
    t = np.arange(int(fs * 0.02)) / fs
    rng = np.random.default_rng(0)
    sig = 0.1 * np.exp(2j * np.pi * 300e3 * t) + 1e-3 * (
        rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))
    )
    path = tmp_path / "cap.cf32"
    save_iq_file(
        path,
        sig.astype(np.complex64),
        sample_rate=fs,
        center_frequency=100e6,
        sample_format=SampleFormat.FLOAT32,
        file_format=FileFormat.RAW,
    )
    rc = main(
        [
            "scan",
            "--input",
            str(path),
            "--sample-rate",
            str(fs),
            "--center",
            "100e6",
            "--threshold",
            "auto",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Detected 1 signal(s)" in out
    assert "100.3000 MHz" in out


def test_real_rtlsdr_driver_is_opened_and_read(monkeypatch, capsys, no_settle):
    """Through the real RTLSDRDevice and DeviceManager, with pyrtlsdr faked."""
    counts = {"open": 0, "read": 0}

    class RtlSdr:
        def __init__(self, device_index=0):
            counts["open"] += 1
            self.sample_rate, self.center_freq, self.gain = 2.4e6, 100e6, "auto"
            self._rng = np.random.default_rng(1)

        @staticmethod
        def get_device_serial_addresses():
            return ["00000001"]

        def set_manual_gain_enabled(self, *_args):
            pass

        def get_gains(self):
            return [0, 100, 200]

        def read_samples(self, n):
            counts["read"] += 1
            t = np.arange(n) / self.sample_rate
            x = (
                self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
            ) * 1e-4
            offset = 100.1e6 - self.center_freq
            if abs(offset) < self.sample_rate / 2:
                x = x + 0.1 * np.exp(2j * np.pi * offset * t)
            return x.astype(np.complex64)

        def close(self):
            pass

    class _Lib:
        @staticmethod
        def rtlsdr_get_device_count():
            return 1

    fake = types.ModuleType("rtlsdr")
    fake.RtlSdr, fake.librtlsdr = RtlSdr, _Lib
    monkeypatch.setitem(sys.modules, "rtlsdr", fake)
    with mock.patch(
        "sdr_module.devices.hackrf.HackRFDevice.list_devices", return_value=[]
    ):
        rc = main(["scan", "--start", "99", "--end", "101", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "100.1000 MHz" in out
    assert counts["open"] == 1
    assert counts["read"] >= 2

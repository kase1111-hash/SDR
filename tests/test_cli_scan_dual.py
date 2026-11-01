"""``sdr-scan scan --dual``: two connected SDRs sweep a range at once (K-X1).

Before, ``sdr-scan scan`` used one device (the first found, or
``--device``'s kind) and there was no way to scan with two together. Now
``--dual`` opens the first two found: each sweeps part of the range (what
only it tunes, then they meet in the middle of the rest, the faster taking
more), and the signals are listed as found and, at the end, in frequency
order, each once.

The devices are fakes behind a fake DeviceManager (``FakeTuner``,
``tests/tuner_fakes.py``); one test drives the real RTL-SDR and HackRF One
drivers through fake ``rtlsdr`` and ``python_hackrf`` modules.
"""

import re
import sys
import time
import types
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from sdr_module import cli
from sdr_module.cli import main
from tests.tuner_fakes import STATIONS, FakeTuner


class _RtlKind:
    pass


class _HackRFKind:
    pass


@dataclass
class _Info:
    name: str
    index: int = 0


@dataclass
class _Detected:
    info: _Info
    device_class: type
    is_available: bool = True


class DualManager:
    """A DeviceManager that finds and opens the devices given."""

    DEVICE_TYPES = {"rtlsdr": _RtlKind, "hackrf": _HackRFKind}

    def __init__(self, *devices):
        # (kind, index, device): device.info.name is what it is called.
        self.devices = list(devices)
        self.opened: List[tuple] = []
        self.closed = 0

    def scan_devices(self):
        kinds = {"rtlsdr": _RtlKind, "hackrf": _HackRFKind}
        return [
            _Detected(_Info(device.info.name, index), kinds[kind])
            for kind, index, device in self.devices
        ]

    def open_device(self, device_type, index=0, config=None):
        self.opened.append((device_type, index))
        for kind, i, device in self.devices:
            if (kind, i) == (device_type, index):
                return device
        return None

    def close_all(self):
        self.closed += 1


def rtl(name="RTL-SDR #0", **options):
    options.setdefault("low_hz", 24e6)
    options.setdefault("high_hz", 1766e6)
    return FakeTuner(name, **options)


def hackrf(name="HackRF One", **options):
    options.setdefault("low_hz", 1e6)
    options.setdefault("high_hz", 6e9)
    return FakeTuner(name, **options)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(cli, "_LIVE_SETTLE_S", 0.0)
    monkeypatch.setattr(cli, "_LIVE_READ_TIMEOUT_S", 0.2)


def _use(monkeypatch, manager):
    from sdr_module.core import device_manager

    monkeypatch.setattr(device_manager, "DeviceManager", lambda: manager)
    return manager


def _listed(out):
    """The frequencies (MHz) of the final list, in the order printed."""
    final = out.split("in frequency order:")[-1]
    return [float(m) for m in re.findall(r"^\s+(\d+\.\d{4}) MHz", final, re.M)]


def test_two_devices_sweep_the_range_together(monkeypatch, capsys, fast):
    a, b = rtl(), hackrf()
    manager = _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("hackrf", 0, b)))
    rc = main(["scan", "--start", "88", "--end", "108", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "with RTL-SDR #0 (2.4 MS/s) and HackRF One (2.4 MS/s) at once" in out
    assert "RTL-SDR #0 (2.4 MS/s): from 88.000 MHz up" in out
    assert "HackRF One (2.4 MS/s): from 108.000 MHz down" in out
    # Each station once as found (with the receiver that found it), then
    # all of them in frequency order.
    for freq in STATIONS:
        assert out.count(f"{freq / 1e6:10.4f} MHz") == 2, out
    assert _listed(out) == sorted(f / 1e6 for f in STATIONS)
    assert "4 signal(s) in frequency order:" in out
    assert re.search(r"Sweep 1 done in [\d.]+ s: 4 signal\(s\), 4 new\.", out)
    assert re.search(r"RTL-SDR #0 swept 88\.000 - [\d.]+ MHz \(\d+ steps?\)", out)
    assert re.search(r"HackRF One [\d.]+ - 108\.000 MHz \(\d+ steps?\)", out)
    # Both swept, each its own part; both opened, started, stopped, closed.
    assert a.tuned and b.tuned and max(a.tuned) < min(b.tuned)
    assert manager.opened == [("rtlsdr", 0), ("hackrf", 0)]
    assert (a.started, a.stopped, b.started, b.stopped) == (1, 1, 1, 1)
    assert manager.closed == 1


def test_one_rate_each(monkeypatch, capsys, fast):
    a, b = rtl(), hackrf()
    _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("hackrf", 0, b)))
    rc = main(
        ["scan", "--start", "88", "--end", "108", "--dual", "--single"]
        + ["--rate", "2.4e6,10e6"]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert (a.sample_rate, b.sample_rate) == (2.4e6, 10e6)
    assert "HackRF One (10 MS/s)" in out
    assert _listed(out) == sorted(f / 1e6 for f in STATIONS)


def test_two_rates_need_dual(monkeypatch, capsys):
    _use(monkeypatch, DualManager(("rtlsdr", 0, rtl())))
    assert main(["scan", "--single", "--rate", "2.4e6,10e6"]) == 1
    assert "two rates are for --dual" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["scan", "--rate", "fast"])


def test_each_device_sweeps_only_what_it_tunes(monkeypatch, capsys, fast):
    stations = {1720e6: -30.0, 1850e6: -30.0}
    a, b = rtl(stations=stations), hackrf(stations=stations)
    _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("hackrf", 0, b)))
    rc = main(["scan", "--start", "1700", "--end", "1900", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "1766.100 - 1900.000 MHz (only it tunes those)" in out
    assert _listed(out) == [1720.0, 1850.0]
    assert a.tuned_inside(24e6, 1766e6)


def test_a_range_neither_tunes_is_refused(monkeypatch, capsys):
    _use(
        monkeypatch, DualManager(("rtlsdr", 0, rtl()), ("rtlsdr", 1, rtl("RTL-SDR #1")))
    )
    rc = main(["scan", "--start", "2000", "--end", "2100", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "neither receiver tunes 2000.000 - 2100.000 MHz" in out
    assert "RTL-SDR #0 tunes 24.000 - 1766.000 MHz" in out
    assert "choose a range inside what they tune" in out


def test_the_range_is_limited_to_what_they_tune(monkeypatch, capsys, fast):
    a, b = rtl("RTL-SDR #0"), rtl("RTL-SDR #1")
    _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("rtlsdr", 1, b)))
    rc = main(["scan", "--start", "1760", "--end", "1800", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "limited to what RTL-SDR #0 and RTL-SDR #1 tune" in out
    assert a.tuned_inside(24e6, 1766e6) and b.tuned_inside(24e6, 1766e6)


def test_without_two_devices_it_says_what_to_do(monkeypatch, capsys):
    _use(monkeypatch, DualManager(("rtlsdr", 0, rtl())))
    rc = main(["scan", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "--dual needs two connected SDRs, but only RTL-SDR #0 is" in out
    assert "without --dual" in out
    _use(monkeypatch, DualManager())
    assert main(["scan", "--dual", "--single"]) == 1
    assert "No SDR device is connected" in capsys.readouterr().out


def test_with_device_both_are_of_that_kind(monkeypatch, capsys, fast):
    manager = _use(
        monkeypatch,
        DualManager(
            ("rtlsdr", 0, rtl()),
            ("hackrf", 0, hackrf()),
            ("rtlsdr", 1, rtl("RTL-SDR #1")),
        ),
    )
    rc = main(
        ["scan", "--start", "99", "--end", "101", "--dual", "--single"]
        + ["--device", "rtlsdr"]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert manager.opened == [("rtlsdr", 0), ("rtlsdr", 1)]
    assert "RTL-SDR #0" in out and "RTL-SDR #1" in out


def test_a_silent_device_leaves_its_part_to_the_other(monkeypatch, capsys, fast):
    a, b = rtl(), hackrf(deliver=False)
    _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("hackrf", 0, b)))
    rc = main(["scan", "--start", "88", "--end", "108", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "HackRF One swept nothing." in out
    assert "Note: HackRF One returned no samples" in out
    assert _listed(out) == sorted(f / 1e6 for f in STATIONS)


def test_neither_device_delivering_is_an_error(monkeypatch, capsys, fast):
    a, b = rtl(deliver=False), hackrf(deliver=False)
    manager = _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("hackrf", 0, b)))
    rc = main(["scan", "--start", "100", "--end", "101", "--dual", "--single"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error: No receiver returned samples" in out
    assert (a.stopped, b.stopped, manager.closed) == (1, 1, 1)


def test_continuous_until_ctrl_c(monkeypatch, capsys, fast):
    a, b = rtl(), hackrf()
    manager = _use(monkeypatch, DualManager(("rtlsdr", 0, a), ("hackrf", 0, b)))
    said = cli._say_dual_pass

    def say_then_stop(*args, **kwargs):
        said(*args, **kwargs)
        if args[0].number == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_say_dual_pass", say_then_stop)
    rc = main(["scan", "--start", "88", "--end", "108", "--dual"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Press Ctrl+C to stop." in out
    assert "Sweep 1 done" in out and "Sweep 2 done" in out and "Stopped." in out
    # Each station listed once as found, however many sweeps saw it.
    assert out.split("Stopped.")[0].count("100.1000 MHz") == 1
    assert _listed(out) == sorted(f / 1e6 for f in STATIONS)
    assert (a.stopped, b.stopped, manager.closed) == (1, 1, 1)


def test_through_the_real_drivers(monkeypatch, capsys, fast):
    """An RTL-SDR (fake pyrtlsdr) and a HackRF One (fake python_hackrf),
    each opened and read through its real driver."""
    from sdr_module.core import device_manager
    from tests.hackrf_fakes import ThreadedFakeHackRF, install_fake_backend

    # (A transfer in flight at a retune holds the old frequency, as in
    # hardware: the settle time drops it.)
    monkeypatch.setattr(cli, "_LIVE_SETTLE_S", 0.15)
    monkeypatch.setattr(cli, "_LIVE_READ_TIMEOUT_S", 1.0)
    stations = {99.5e6: -30.0, 100.5e6: -25.0}
    reads = {"rtl": 0}

    class RtlSdr:
        def __init__(self, device_index=0):
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
            # (In real time, made on the frequency tuned when handed over.)
            time.sleep(n / self.sample_rate)
            reads["rtl"] += 1
            t = np.arange(n) / self.sample_rate
            x = (
                self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
            ) * 1e-3
            for freq, level in stations.items():
                offset = freq - self.center_freq
                if abs(offset) < self.sample_rate / 2:
                    x = x + 10 ** (level / 20) * np.exp(2j * np.pi * offset * t)
            return x.astype(np.complex64)

        def close(self):
            pass

    class _Lib:
        @staticmethod
        def rtlsdr_get_device_count():
            return 1

    fake_rtl = types.ModuleType("rtlsdr")
    fake_rtl.RtlSdr, fake_rtl.librtlsdr = RtlSdr, _Lib
    monkeypatch.setitem(sys.modules, "rtlsdr", fake_rtl)
    fake_hackrf = ThreadedFakeHackRF(rx_stations=stations)
    install_fake_backend(monkeypatch, fake_hackrf)
    assert device_manager.DeviceManager is not None
    rc = main(
        ["scan", "--start", "98", "--end", "102", "--dual", "--single"]
        + ["--rate", "2.4e6,10e6"]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "with RTL-SDR #0 (2.4 MS/s) and HackRF One (10 MS/s) at once" in out
    assert _listed(out) == [99.5, 100.5]
    assert reads["rtl"] >= 1 and fake_hackrf.rx_transfers >= 1
    assert fake_hackrf.closed

"""The dual-SDR wideband scan (``sdr_module.core.wideband_scan``, K-X1).

Before, wideband scanning was not implemented: ``DualSDRController`` refused
``OperationMode.WIDEBAND_SCAN`` and there was no way to have two devices
sweep one range together. Now two devices sweep complementary parts of the
range at once, each inside what it tunes and in steps sized for its own
sample rate, and the signals they find are merged, each once, in frequency
order, with progress and cancel.

The devices are fakes (``tests/tuner_fakes.py``) that hear carriers at
known frequencies around wherever they are tuned; one test runs two demo
devices in real time through ``PolledDevice``.
"""

import threading
import time

import numpy as np
import pytest

from sdr_module.core.wideband_scan import (
    CANCELLED,
    COMPLETE,
    NO_SAMPLES,
    SignalList,
    WidebandHit,
    WidebandScan,
    device_range,
    device_rate,
)
from sdr_module.dsp.sweep import ChannelReading
from tests.tuner_fakes import STATIONS, FakeTuner

RTL_RANGE = dict(low_hz=24e6, high_hz=1766e6)
HACKRF_RANGE = dict(low_hz=1e6, high_hz=6e9)


def _scan(devices, start=88e6, end=108e6, step=100e3, **options):
    options.setdefault("settle_s", 0.0)
    options.setdefault("read_timeout_s", 0.5)
    return WidebandScan(devices, start, end, step, **options)


def _freqs(hits):
    return [round(h.frequency_hz) for h in hits]


def _most_queued(scan, device):
    """Run ``scan``; the most transfers ``device`` held queued meanwhile
    (looked at every 10 ms)."""
    queued = []
    done = threading.Event()

    def watch():
        while not done.wait(0.01):
            queued.append(device._sample_queue.qsize())

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        scan.run()
    finally:
        done.set()
        watcher.join(5.0)
    assert queued
    return max(queued)


def _no_scan_threads():
    """No device thread of a scan is left (the background scan's own thread
    may still be returning after wait())."""
    names = [t.name for t in threading.enumerate()]
    return not any(n.startswith("wideband-") and n != "wideband-scan" for n in names)


class TestTwoDevices:
    def test_complementary_parts_merged_in_frequency_order(self):
        a = FakeTuner("A", **RTL_RANGE)
        b = FakeTuner("B", **HACKRF_RANGE)
        passes = []
        scan = _scan([a, b], on_pass=passes.append)
        hits = scan.run()
        # Every station once, in frequency order.
        assert _freqs(hits) == sorted(STATIONS)
        assert scan.reason == COMPLETE
        # Both swept, each its own part: A from the bottom up, B from the
        # top down, so all of A's steps lie below all of B's.
        assert a.tuned and b.tuned
        assert max(a.tuned) < min(b.tuned)
        report = passes[0]
        assert report.number == 1 and report.found == len(STATIONS)
        assert report.steps[0] == len(a.tuned) and report.steps[1] == len(b.tuned)
        ((a_low, a_high),) = report.covered[0]
        ((b_low, b_high),) = report.covered[1]
        assert a_low == pytest.approx(88e6) and b_high == pytest.approx(108e6)
        assert b_low == pytest.approx(a_high + 100e3)  # they met
        assert report.missed == () and report.stopped == ("", "")
        # Each found by the one sweeping its part.
        for hit in hits:
            part = report.covered[hit.receiver][0]
            assert part[0] - 50e3 <= hit.frequency_hz <= part[1] + 50e3
            assert hit.receiver_name == ("A", "B")[hit.receiver]
        # Streams started and stopped once; no thread left behind.
        assert (a.started, a.stopped, b.started, b.stopped) == (1, 1, 1, 1)
        assert _no_scan_threads()

    def test_each_device_stays_inside_what_it_tunes(self):
        """1700-1900 MHz: the RTL-SDR stops at 1766 MHz, the HackRF takes
        everything above it (and shares the rest)."""
        stations = {1720e6: -30.0, 1850e6: -30.0}
        a = FakeTuner("RTL", stations=stations, **RTL_RANGE)
        b = FakeTuner("HackRF", stations=stations, **HACKRF_RANGE)
        hits = _scan([a, b], 1700e6, 1900e6, 100e3).run()
        assert _freqs(hits) == [1720e6, 1850e6]
        assert a.tuned and a.tuned_inside(24e6, 1766e6)
        assert {h.frequency_hz: h.receiver for h in hits}[1850e6] == 1

    def test_steps_follow_each_rate(self):
        a = FakeTuner("A", rate=2.4e6)
        b = FakeTuner("B", rate=10e6)
        scan = _scan([a, b], 88e6, 148e6)
        assert scan.rates == (2.4e6, 10e6)
        pages = {0: [], 1: []}
        claim = scan.sweep.claim

        def claiming(receiver):
            page = claim(receiver)
            if page is not None:
                pages[receiver].append(page)
            return page

        scan.sweep.claim = claiming
        hits = scan.run()
        assert _freqs(hits) == sorted(STATIONS)
        # A full page holds 19 channels (1.9 MHz) at 2.4 MS/s and 79 (7.9
        # MHz) at 10 MS/s (the page where they meet may hold fewer); each
        # device tuned its pages' LOs, A going up and B down.
        assert pages[0][0].count == 19 and max(p.count for p in pages[0]) == 19
        assert pages[1][0].count == 79 and max(p.count for p in pages[1]) == 79
        assert a.tuned == [p.lo_hz for p in pages[0]]
        assert b.tuned == [p.lo_hz for p in pages[1]]
        assert np.all(np.diff(a.tuned) > 0) and np.all(np.diff(b.tuned) < 0)

    def test_samples_from_before_a_retune_are_dropped(self):
        """A transfer in flight at a retune holds the old frequency: after
        the settle time it is thrown away, not measured (it would put the
        last step's stations at this step's frequencies)."""

        def run(settle_s):
            devices = [
                FakeTuner("A", in_flight_s=0.03),
                FakeTuner("B", in_flight_s=0.03),
            ]
            return _freqs(_scan(devices, settle_s=settle_s).run())

        assert run(0.06) == sorted(STATIONS)
        # (Without the settle time, phantoms: the fakes do hold the old
        # frequency's samples.)
        assert len(run(0.0)) > len(STATIONS)

    def test_one_device_scans_alone(self):
        a = FakeTuner("A")
        scan = _scan([a])
        assert _freqs(scan.run()) == sorted(STATIONS)
        assert scan.names == ("A",)

    def test_a_device_done_with_its_part_is_not_left_queuing(self, monkeypatch):
        """A device that has measured its part streams on until the pass
        ends (here 88-90 MHz is all it tunes; the other has 90.1-140 MHz to
        itself). What it delivers meanwhile is thrown away as it comes:
        before, it filled the device's queue (100 transfers, 400 MB from an
        RTL-SDR, whose pyrtlsdr hands over complex128) and then dropped
        samples until the other device had finished."""
        from sdr_module.devices.base import DeviceSpec
        from sdr_module.devices.polled import PolledDevice
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)
        demo = MockDevice()
        demo.spec = DeviceSpec(88e6, 90e6, 1e6, 3.2e6, 3.2e6, 8, 0.0, 50.0, 10.0)
        quick = PolledDevice(demo, "Quick")  # (10 ms transfers)
        slow = FakeTuner("Slow", **HACKRF_RANGE)
        scan = _scan([quick, slow], 88e6, 140e6, settle_s=0.03)
        most = _most_queued(scan, quick)
        report = scan.passes[0]
        # Quick measured its two steps long before Slow was done.
        assert report.steps[0] == 2 and report.steps[1] >= 27
        # At most a few transfers (0.2 s) waited at any time.
        assert most < 20, most
        assert quick.dropped_samples == 0

    def test_a_device_waiting_for_the_others_last_step_is_not_left_queuing(
        self, monkeypatch
    ):
        """At the end of a pass a device waits while the other measures a
        step it could still give back: it drops what it delivers meanwhile
        (before, its queue filled for as long as the other took)."""
        from sdr_module.devices.polled import PolledDevice
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)
        quick = PolledDevice(MockDevice(), "Quick")  # (10 ms transfers)
        slow = FakeTuner("Slow")
        read = slow.read_samples

        def slow_read(n, timeout=1.0):
            if timeout:  # (a measurement, not a drain)
                time.sleep(0.6)
            return read(n, timeout)

        slow.read_samples = slow_read
        scan = _scan([quick, slow], 88e6, 92e6)
        most = _most_queued(scan, quick)
        # Quick took 88.0-90.1 MHz in two steps, then waited about 0.6 s
        # for Slow's one step (90.2-92.0 MHz).
        assert scan.passes[0].steps == (2, 1)
        assert most < 20, most


class TestProgressAndCancel:
    def test_progress_rises_to_one(self):
        seen = []
        scan = _scan([FakeTuner("A"), FakeTuner("B")], on_progress=seen.append)
        scan.run()
        fractions = [p.fraction for p in seen]
        assert fractions == sorted(fractions)
        assert fractions[-1] == pytest.approx(1.0)
        assert sum(seen[-1].steps) == len(seen)
        assert seen[-1].found == len(STATIONS)
        assert scan.progress == pytest.approx(1.0)

    def test_cancel_from_a_callback(self):
        a, b = FakeTuner("A"), FakeTuner("B")
        scan = _scan([a, b], 24e6, 1766e6)
        scan.on_progress = lambda progress: scan.cancel()
        scan.run()
        assert scan.reason == CANCELLED
        # Stopped at once: of the range's 917 steps, each device finished
        # or dropped the one it was on (a few at most, by the time the
        # first was told).
        (report,) = scan.passes
        assert not report.complete and sum(report.steps) < 50
        assert len(a.tuned) + len(b.tuned) < 50
        assert (a.stopped, b.stopped) == (1, 1)
        assert _no_scan_threads()

    def test_cancel_while_a_device_waits_for_samples(self):
        """Cancel ends the scan promptly even while a read is waiting."""
        a = FakeTuner("A", deliver=False)
        scan = _scan([a], read_timeout_s=60.0)
        scan.start()
        assert a.reading.wait(5.0)
        scan.cancel()
        assert scan.wait(5.0)
        assert scan.reason == CANCELLED
        assert not scan.running and a.stopped == 1
        assert _no_scan_threads()

    def test_continuous_lists_each_signal_once(self):
        found, passes = [], []
        scan = _scan(
            [FakeTuner("A"), FakeTuner("B")],
            continuous=True,
            on_found=found.append,
            on_pass=passes.append,
        )

        def on_pass(report):
            passes.append(report)
            if report.number == 2:
                scan.cancel()

        scan.on_pass = on_pass
        hits = scan.run()
        assert scan.reason == CANCELLED
        assert [p.number for p in passes] == [1, 2]
        assert [p.new for p in passes] == [len(STATIONS), 0]
        assert len(found) == len(STATIONS) and _freqs(hits) == sorted(STATIONS)

    def test_ctrl_c_stops_the_devices(self):
        a, b = FakeTuner("A"), FakeTuner("B")

        def interrupt(hit):
            raise KeyboardInterrupt

        scan = _scan([a, b], on_found=interrupt)
        with pytest.raises(KeyboardInterrupt):
            scan.run()
        assert scan.reason == CANCELLED and not scan.running
        assert (a.stopped, b.stopped) == (1, 1)
        assert _no_scan_threads()

    def test_background_scan(self):
        threads = []
        scan = _scan(
            [FakeTuner("A"), FakeTuner("B")],
            on_found=lambda hit: threads.append(threading.current_thread()),
        )
        finished = []
        scan.on_finished = finished.append
        scan.start()
        assert scan.wait(20.0)
        assert finished == [COMPLETE]
        assert _freqs(scan.hits) == sorted(STATIONS)
        assert threads and threading.main_thread() not in threads
        with pytest.raises(RuntimeError):
            scan.run()  # (a scan runs once)

    def test_a_background_scan_without_a_thread_has_ended(self, monkeypatch):
        """start() raises when no thread can be had; the scan then says it
        isn't running and why, and wait() returns (it waited for ever)."""
        a = FakeTuner("A")
        scan = _scan([a])

        def no_thread(self):
            raise RuntimeError("can't start new thread")

        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", no_thread)
            with pytest.raises(RuntimeError):
                scan.start()
        assert not scan.running
        assert scan.wait(0)
        assert "can't start new thread" in scan.reason
        assert a.started == 0


class TestFailures:
    def test_a_device_without_samples_hands_its_part_to_the_other(self):
        a = FakeTuner("A")
        b = FakeTuner("B", deliver=False)
        passes = []
        scan = _scan([a, b], read_timeout_s=0.05, on_pass=passes.append)
        hits = scan.run()
        assert _freqs(hits) == sorted(STATIONS)
        assert scan.reason == COMPLETE
        report = passes[0]
        assert report.steps[1] == 0 and report.failed[1] >= 1
        assert "returned no samples" in report.stopped[1]
        ((low, high),) = report.covered[0]
        assert low == pytest.approx(88e6) and high == pytest.approx(108e6)
        assert report.missed == ()

    def test_what_only_the_silent_device_tunes_is_missed(self):
        stations = {1720e6: -30.0, 1850e6: -30.0}
        a = FakeTuner("RTL", stations=stations, **RTL_RANGE)
        b = FakeTuner("HackRF", stations=stations, deliver=False, **HACKRF_RANGE)
        passes = []
        hits = _scan(
            [a, b], 1700e6, 1900e6, read_timeout_s=0.05, on_pass=passes.append
        ).run()
        assert _freqs(hits) == [1720e6]
        ((low, high),) = passes[0].missed
        assert low == pytest.approx(1766.1e6) and high == pytest.approx(1900e6)

    def test_nothing_from_either_device_is_an_error(self):
        scan = _scan(
            [FakeTuner("A", deliver=False), FakeTuner("B", deliver=False)],
            read_timeout_s=0.02,
        )
        assert scan.run() == []
        assert scan.reason == NO_SAMPLES

    def test_a_device_that_does_not_start_leaves_the_scan_to_the_other(self):
        a, b = FakeTuner("A"), FakeTuner("B")
        b.start_rx = lambda callback=None: False
        passes = []
        hits = _scan([a, b], on_pass=passes.append).run()
        assert _freqs(hits) == sorted(STATIONS)
        assert "did not start" in passes[0].stopped[1]
        assert b.tuned == [] and b.stopped == 0

    def test_a_device_already_streaming_is_left_alone(self):
        """One someone else is reading (dual receive, a callback) is not
        read, retuned or stopped by the scan."""
        a, b = FakeTuner("A"), FakeTuner("B")
        b.streaming = True
        passes = []
        hits = _scan([a, b], on_pass=passes.append).run()
        assert _freqs(hits) == sorted(STATIONS)
        assert "already streaming" in passes[0].stopped[1]
        assert b.tuned == [] and b.stopped == 0 and b.streaming

    def test_nothing_is_left_queued_after_the_scan(self, monkeypatch):
        from sdr_module.devices.polled import PolledDevice
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)
        device = PolledDevice(MockDevice(), "Demo")
        WidebandScan([device], 100e6, 100.4e6, 100e3, settle_s=0.0).run()
        assert device.read_samples(16, timeout=0.0) is None

    def test_a_device_that_raises_is_left_out(self):
        a, b = FakeTuner("A"), FakeTuner("B")

        def broken(n, timeout=1.0):
            raise OSError("unplugged")

        b.read_samples = broken
        passes = []
        hits = _scan([a, b], on_pass=passes.append).run()
        assert _freqs(hits) == sorted(STATIONS)
        assert "unplugged" in passes[0].stopped[1]


class TestArguments:
    def test_bad_scans_are_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="needs a device"):
            WidebandScan([], 88e6, 108e6, 100e3)
        with pytest.raises(ValueError, match="one or two"):
            WidebandScan([FakeTuner(), FakeTuner(), FakeTuner()], 88e6, 108e6, 1e5)
        with pytest.raises(ValueError, match="above its start"):
            WidebandScan([FakeTuner()], 108e6, 88e6, 100e3)
        with pytest.raises(ValueError, match="sample rate"):
            WidebandScan([FakeTuner(rate=0.0)], 88e6, 108e6, 100e3)
        with pytest.raises(ValueError, match="1766.000"):
            WidebandScan(
                [FakeTuner("RTL", **RTL_RANGE), FakeTuner("RTL 2", **RTL_RANGE)],
                2000e6,
                2100e6,
                100e3,
            )

    def test_same_names_are_told_apart(self):
        scan = WidebandScan([FakeTuner("Demo"), FakeTuner("Demo")], 88e6, 108e6, 1e5)
        assert scan.names == ("Demo 1", "Demo 2")

    def test_device_rate_and_range(self):
        from sdr_module.devices.base import DeviceSpec
        from sdr_module.devices.polled import PolledDevice
        from sdr_module.gui.device_dialog import MockDevice

        demo = PolledDevice(MockDevice(), "Demo")
        assert device_rate(demo) == 2.4e6
        assert device_range(demo) == (0.0, float("inf"))
        tuner = FakeTuner(rate=3.2e6, **RTL_RANGE)
        assert device_rate(tuner) == 3.2e6
        assert device_range(tuner) == (24e6, 1766e6)
        # A device without ``state`` (the CLI's fakes): its sample_rate.
        bare = type("Bare", (), {"sample_rate": 1e6})()
        assert device_rate(bare) == 1e6
        assert device_rate(type("Unknown", (), {})()) == 0.0
        tuner.spec = DeviceSpec(1e6, 6e9, 2e6, 20e6, 20e6, 8, 0, 62, -5)
        assert device_range(tuner) == (1e6, 6e9)
        tuner.spec = type("Spec", (), {"freq_min": float("nan"), "freq_max": 5})()
        assert device_range(tuner) == (0.0, float("inf"))


class TestSignalList:
    def _reading(self, freq, peak, width=10e3):
        return ChannelReading(freq, peak, -90.0, width)

    def test_one_entry_per_signal_the_strongest_kept(self):
        signals = SignalList(88e6, 108e6, 100e3)
        first = signals.add(self._reading(100.1e6, -30.0, 150e3), 0, "A")
        assert isinstance(first, WidebandHit) and first.receiver == 0
        # The same station seen by the other device across the seam.
        assert signals.add(self._reading(100.12e6, -25.0, 150e3), 1, "B") is None
        assert signals.add(self._reading(100.08e6, -35.0, 150e3), 0, "A") is None
        other = signals.add(self._reading(93.3e6, -40.0), 1, "B")
        assert other is not None
        hits = signals.hits()
        assert [h.frequency_hz for h in hits] == [93.3e6, 100.12e6]
        assert hits[1].receiver_name == "B" and hits[1].peak_db == -25.0
        assert len(signals) == 2

    def test_outside_the_range_is_left_out(self):
        signals = SignalList(88e6, 108e6, 100e3)
        assert signals.add(self._reading(87.9e6, -20.0), 0, "A") is None
        assert signals.add(self._reading(87.96e6, -20.0), 0, "A") is not None
        assert len(signals) == 1


class TestDemoDevices:
    def test_two_demo_devices_find_the_demo_station(self, monkeypatch):
        """Two demo devices (``MockDevice`` through ``PolledDevice``, read
        in real time) sweep the FM band together and find the demo's
        strongest station."""
        from sdr_module.devices.polled import PolledDevice
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)
        first, second = MockDevice(), MockDevice()
        second.set_sample_rate(3.2e6)
        devices = [PolledDevice(first, "Demo A"), PolledDevice(second, "Demo B")]
        started = time.monotonic()
        scan = WidebandScan(devices, 96e6, 104e6, 100e3, settle_s=0.03)
        hits = scan.run()
        assert time.monotonic() - started < 20
        assert scan.reason == COMPLETE
        assert 100.1e6 in _freqs(hits)
        assert all(d.state.is_streaming is False for d in devices)
        assert scan.passes[0].steps[0] and scan.passes[0].steps[1]

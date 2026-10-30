#!/usr/bin/env python3
"""
Main window display loop and audio: fixes from the code review.

An I/Q recording keeps every sample while this computer can't process the
stream in real time (and says when the driver lost some), the display
pacer recovers from a passing overload, Balance I/Q survives a block with
no I/Q variance, the audio sink tops up the output between writes, the
RDS badge forgets the station when the source changes, an empty channel
block hands no stale audio on, and the TONE state is judged whether the
speaker plays or not.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_review_mainloop.py``
"""

import collections
import os
import queue
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)
from tests.test_gui_round3_main import _FakeClock, _FakeDemo  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


_saved_style = None


def setUpModule():  # noqa: N802 - unittest hook
    """Wiring tests: run on the unstyled app (much faster to build)."""
    global _saved_style
    if not HAS_PYQT6:
        return
    qapp = QApplication.instance()
    _saved_style = (qapp.styleSheet(), qapp.palette())
    qapp.setStyleSheet("")


def tearDownModule():  # noqa: N802 - unittest hook
    if not HAS_PYQT6 or _saved_style is None:
        return
    qapp = QApplication.instance()
    qapp.setStyleSheet(_saved_style[0])
    qapp.setPalette(_saved_style[1])


def _mw():
    from sdr_module.gui import main_window

    return main_window


TRANSFER = 131072  # one HackRF USB transfer


class _Clock(_FakeClock):
    """The fake ``time`` module, with what a recording also asks it."""

    @staticmethod
    def gmtime(*args):
        import time

        return time.gmtime(*args)


def _queued_hardware(rate=20e6):
    """A hardware-like SDRDevice: :meth:`deliver` plays the driver's USB
    thread, queueing transfers in the base class's queue (100 deep) and
    dropping (and counting) one that doesn't fit, as the drivers do.
    Sample 0 of each transfer holds its running number."""
    from sdr_module.devices.base import SDRDevice

    class QueuedHardware(SDRDevice):
        def __init__(self):
            super().__init__()
            self._state.sample_rate = rate
            self._state.is_streaming = True
            self.sent = 0
            self._dropped_samples = 0

        def deliver(self, transfers):
            for _ in range(transfers):
                block = np.full(TRANSFER, 0.01, dtype=np.complex64)
                block[0] = self.sent
                self.sent += 1
                try:
                    self._sample_queue.put_nowait(block)
                except queue.Full:
                    self._dropped_samples += TRANSFER

        @property
        def dropped_samples(self):
            return self._dropped_samples

        def open(self, index=0):
            return True

        def close(self):
            pass

        def set_frequency(self, freq_hz):
            return True

        def set_sample_rate(self, rate_hz):
            self._state.sample_rate = rate_hz
            return True

        def set_bandwidth(self, bw_hz):
            return True

        def set_gain(self, gain_db):
            return True

        def set_gain_mode(self, auto):
            return True

        def start_rx(self, callback=None):
            return True

        def stop_rx(self):
            return True

    return QueuedHardware()


# ---------------------------------------------------------------------------
# mainloop-recording-drops-samples-overloaded
# ---------------------------------------------------------------------------


class _HardwareCase(_WindowTestCase):
    """The window on a hardware-like device at 20 MS/s, with a fake clock
    and processing that costs ``LOAD`` times real time."""

    RATE = 20e6
    LOAD = 1.5

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-review-mainloop-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = self.folder
        self.clock = _Clock()
        self.dev = _queued_hardware(self.RATE)
        self.win._device = self.dev
        self.win._is_running = True
        self.win._pacer.reset()
        self.win._rx_history.clear()
        process = self.win._process_block

        def slow_process(samples):
            self.clock.advance(self.LOAD * len(samples) / self.RATE)
            process(samples)

        self.win._process_block = slow_process
        patcher = mock.patch.object(self.mw, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if HAS_PYQT6:
            if self.win._recording:
                self.win._toggle_recording(False)
            self.win._is_running = False
            self.win._device = None
        super().tearDown()

    def run_frames(self, frames, per_frame=5):
        """``frames`` display frames, the driver queueing ``per_frame``
        transfers before each (20 MS/s at 30 frames a second)."""
        for _ in range(frames):
            self.dev.deliver(per_frame)
            tick = self.clock.now
            self.win._update_display()
            self.clock.now = max(self.clock.now, tick + 0.033)  # the timer

    def recorded_transfers(self):
        data = np.concatenate(list(self.win._samples_buffer))
        self.assertEqual(len(data) % TRANSFER, 0)
        return np.rint(data.reshape(-1, TRANSFER)[:, 0].real).astype(int)


class TestRecordingWhileOverloaded(_HardwareCase):
    def test_every_transfer_is_recorded_while_frames_cannot_keep_up(self):
        self.run_frames(40)
        self.assertTrue(self.win._pacer.overloaded)  # 1.5x real time
        # (Reading one transfer a frame, the driver's queue is full: it
        # holds transfers from the last seconds, with gaps between them.)
        self.assertGreater(self.dev.dropped_samples, 0)
        first = self.dev.sent
        self.win._toggle_recording(True)
        self.run_frames(1)
        dropped = self.dev.dropped_samples
        self.run_frames(89)  # three seconds
        self.win._toggle_recording(False)
        # The take starts with the frame after Record (not with the stale
        # queue) and holds every transfer the driver delivered from then
        # on, in order, none dropped for want of reading the queue...
        indices = self.recorded_transfers()
        self.assertGreaterEqual(indices[0], first)
        self.assertLessEqual(indices[0], first + 5)
        np.testing.assert_array_equal(indices, np.arange(indices[0], self.dev.sent))
        self.assertEqual(self.dev.dropped_samples, dropped)
        self.assertNotIn("lost", self.win._take_summary()[0])
        # ...while frames still process about one transfer each (the window
        # stays responsive) and the audio stays paused.
        self.assertTrue(self.win._pacer.overloaded)
        self.assertFalse(self.win._realtime())

    def test_frames_that_keep_up_record_and_process_everything(self):
        type(self).LOAD = 0.3
        try:
            seen = []
            process = self.win._process_block
            self.win._process_block = lambda s: (seen.append(len(s)), process(s))
            self.win._toggle_recording(True)
            self.run_frames(60)
            self.win._toggle_recording(False)
        finally:
            type(self).LOAD = 1.5
        self.assertFalse(self.win._pacer.overloaded)
        indices = self.recorded_transfers()
        np.testing.assert_array_equal(indices, np.arange(self.dev.sent))
        self.assertEqual(sum(seen), self.dev.sent * TRANSFER)  # all processed


class TestRecordingGapsAreReported(_HardwareCase):
    LOAD = 0.3

    def test_samples_the_driver_dropped_during_a_take_are_reported(self):
        self.dev._dropped_samples = 40 * TRANSFER  # before the take: not its
        self.run_frames(10)
        self.win._toggle_recording(True)
        self.run_frames(10)
        self.assertNotIn("gaps", self.win._message_label.text())
        # The driver's queue overflowed (the disk or the computer stalled).
        self.dev._dropped_samples += 3 * TRANSFER
        self.run_frames(3)
        text = self.win._message_label.text()
        self.assertIn("gaps", text)
        self.assertIn("20 ms", text)  # 3 transfers at 20 MS/s
        self.assertEqual(self.win._message_label.property("tone"), "warning")
        self.dev._dropped_samples += 2 * TRANSFER
        self.run_frames(3)
        self.win._toggle_recording(False)
        # What was lost stays on the take line once it stopped.
        text, state, tip = self.win._take_summary()
        self.assertEqual(state, "unsaved")
        self.assertIn("33 ms lost", text)
        self.assertIn("gaps", tip)

    def test_drops_while_paused_are_not_the_takes(self):
        self.win._toggle_recording(True)
        self.run_frames(5)
        self.win._on_panel_record_paused(True)
        self.dev._dropped_samples += 10 * TRANSFER
        self.run_frames(5)
        self.win._on_panel_record_paused(False)
        self.run_frames(5)
        self.win._toggle_recording(False)
        text, _state, _tip = self.win._take_summary()
        self.assertNotIn("lost", text)

    def test_a_new_take_starts_without_gaps(self):
        self.win._toggle_recording(True)
        self.run_frames(3)
        self.dev._dropped_samples += TRANSFER
        self.run_frames(3)
        self.win._toggle_recording(False)
        self.assertIn("lost", self.win._take_summary()[0])
        self.win._toggle_recording(True)
        self.run_frames(3)
        self.win._toggle_recording(False)
        self.assertNotIn("lost", self.win._take_summary()[0])


class TestDriversCountDrops(unittest.TestCase):
    def test_base_class_reports_no_drops(self):
        from sdr_module.devices.base import SDRDevice

        self.assertEqual(SDRDevice.dropped_samples.fget(_queued_hardware()), 0)

    def test_hackrf_counts_transfers_its_queue_could_not_take(self):
        import sys

        from sdr_module.devices import hackrf as hackrf_mod
        from tests.test_hackrf_driver import make_fake_backend

        mod, pyhackrf, device = make_fake_backend()
        with (
            mock.patch.dict(
                sys.modules,
                {"python_hackrf": mod, "python_hackrf.pyhackrf": pyhackrf},
            ),
            mock.patch.object(hackrf_mod, "_initialized", False),
        ):
            dev = hackrf_mod.HackRFDevice()
            self.assertTrue(dev.open())
            self.assertTrue(dev.start_rx())
            data = np.zeros(2 * 1000, dtype=np.int8)  # 1000 samples
            for _ in range(dev._sample_queue.maxsize):
                device.feed_rx(data)
            self.assertEqual(dev.dropped_samples, 0)
            device.feed_rx(data)
            device.feed_rx(data)
            self.assertEqual(dev.dropped_samples, 2000)
            dev.stop_rx()
            dev.close()

    def test_rtlsdr_counts_reads_its_queue_could_not_take(self):
        import sys
        import time

        from sdr_module.devices.rtlsdr import RTLSDRDevice
        from tests.test_rtlsdr_driver import make_fake_backend

        rtlsdr, librtlsdr = make_fake_backend()
        with mock.patch.dict(
            sys.modules, {"rtlsdr": rtlsdr, "rtlsdr.librtlsdr": librtlsdr}
        ):
            dev = RTLSDRDevice()
            self.assertTrue(dev.open())
            dev._device._read_data = np.ones(4096, dtype=np.complex128)
            self.assertTrue(dev.start_rx())  # nobody reads the queue
            deadline = time.monotonic() + 5.0
            while dev.dropped_samples == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            dev.stop_rx()
            self.assertGreater(dev.dropped_samples, 0)
            self.assertEqual(dev.dropped_samples % 4096, 0)
            dev.close()


# ---------------------------------------------------------------------------
# mainloop-pacer-stuck-overloaded
# ---------------------------------------------------------------------------


def _simulate(
    source,
    rate,
    load,
    fixed,
    stall_at=None,
    stall_frames=3,
    stall_s=0.3,
    seconds=20.0,
    warm_frames=0,
    judge_from=None,
):
    """The real pacer on a virtual clock. The display timer fires every
    33 ms (or when the last frame ends, if later); a frame of ``n`` samples
    costs ``fixed + load * n / rate`` seconds; ``stall_frames`` frames
    from ``stall_at`` take ``stall_s`` each (another program busy), and the
    first ``warm_frames`` frames 60 ms more (filter designs after Start).
    The source is the demo device (``demo_request``) or a hardware queue of
    100 transfers drained as the window does. Returns the fraction of the
    stream processed from ``judge_from`` (3 s after the stall, or after
    the start) and whether the pacer is overloaded at the end."""
    pacer = _mw()._Pacer()
    t = next_tick = queued_to = 0.0
    fifo = collections.deque()
    after = stalled = frames = 0
    start = judge_from
    if start is None:
        start = (stall_at or 0.0) + 3.0
    while t < seconds:
        t = max(t, next_tick)
        if source == "hw":
            while queued_to + TRANSFER <= rate * t:
                queued_to += TRANSFER
                if len(fifo) < 100:
                    fifo.append(TRANSFER)
            limit = pacer.read_limit(rate, t)
            count = 0
            while fifo and (count == 0 or count < limit):
                count += fifo.popleft()
            if count == 0:
                next_tick = t + 0.033
                continue
        else:
            count = pacer.demo_request(rate, t)
        cost = fixed + load * count / rate
        if frames < warm_frames:
            cost += 0.06
        if stall_at is not None and t >= stall_at and stalled < stall_frames:
            cost, stalled = stall_s, stalled + 1
        frames += 1
        began, t = t, t + cost
        pacer.measured(count, rate, cost)
        if began >= start:
            after += count
        next_tick = began + 0.033
    return after / (rate * (seconds - start)), pacer.overloaded


class TestPacerRecovers(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_a_passing_stall_does_not_overload_a_computer_that_keeps_up(self):
        """Frames at 0.45-0.6x real time (the review measured 0.55-0.63 at
        20 MS/s): a 3-frame stall used to trap the pacer for good."""
        for source, rate, fixed in (("demo", 2.4e6, 0.002), ("hw", 20e6, 0.004)):
            for load in (0.45, 0.55, 0.6):
                with self.subTest(source=source, load=load):
                    processed, overloaded = _simulate(
                        source, rate, load, fixed, stall_at=5.0
                    )
                    self.assertFalse(overloaded)
                    self.assertGreater(processed, 0.99)

    def test_slow_first_frames_after_start_do_not_overload(self):
        for source, rate, fixed in (("demo", 2.4e6, 0.002), ("hw", 20e6, 0.004)):
            with self.subTest(source=source):
                processed, overloaded = _simulate(
                    source, rate, 0.6, fixed, warm_frames=3, judge_from=0.5
                )
                self.assertFalse(overloaded)
                self.assertGreater(processed, 0.99)

    def test_a_computer_that_cannot_keep_up_stays_overloaded(self):
        for source, rate, fixed in (("demo", 2.4e6, 0.002), ("hw", 20e6, 0.004)):
            for load in (0.95, 1.5):
                with self.subTest(source=source, load=load):
                    _processed, overloaded = _simulate(source, rate, load, fixed)
                    self.assertTrue(overloaded)

    def test_one_block_frames_while_overloaded_are_not_judged(self):
        mw = _mw()
        pacer, rate = mw._Pacer(), 20e6
        for _ in range(mw._WARM_UP_FRAMES + 3):
            pacer.measured(TRANSFER * 5, rate, 0.05)  # 1.5x real time
        self.assertTrue(pacer.overloaded)
        self.assertEqual(pacer.read_limit(rate, 10.0), 0.0)  # schedules a probe
        # One transfer a frame costs more than real time (painting the
        # plots is per frame), which says nothing about full frames.
        for _ in range(20):
            self.assertEqual(pacer.read_limit(rate, 10.5), 0.0)
            pacer.measured(TRANSFER, rate, 0.012)
        # The probes keep up: full frames resume.
        now, probes = 11.0, 0
        while pacer.overloaded and probes < 10:
            limit = pacer.read_limit(rate, now)
            self.assertGreater(limit, 0)
            pacer.measured(int(limit), rate, 0.6 * limit / rate)
            probes += 1
            now += 0.033
        self.assertFalse(pacer.overloaded)
        self.assertEqual(probes, 1 + mw._RECOVER_PROBES)

    def test_the_first_probe_after_one_block_frames_is_not_judged(self):
        mw = _mw()
        pacer, rate = mw._Pacer(), 20e6
        for _ in range(mw._WARM_UP_FRAMES + 3):
            pacer.measured(TRANSFER * 5, rate, 0.05)
        pacer.read_limit(rate, 10.0)
        # The first full frame after one-block frames builds its buffers
        # anew (slower); the next ones show the real load.
        costs = iter([1.2] + [0.6] * mw._RECOVER_PROBES)
        now = 11.0
        for _ in range(1 + mw._RECOVER_PROBES):
            limit = pacer.read_limit(rate, now)
            self.assertGreater(limit, 0)
            pacer.measured(int(limit), rate, next(costs) * limit / rate)
            now += 0.033
        self.assertFalse(pacer.overloaded)

    def test_a_slow_probe_waits_for_the_next_one(self):
        mw = _mw()
        pacer, rate = mw._Pacer(), 2.4e6
        for _ in range(mw._WARM_UP_FRAMES + 3):
            pacer.measured(80000, rate, 0.05)
        pacer.read_limit(rate, 10.0)
        for ratio in (0.6, 0.9):  # the second (judged) probe is slow
            limit = pacer.read_limit(rate, 11.0)
            self.assertGreater(limit, 0)
            pacer.measured(int(limit), rate, ratio * limit / rate)
        self.assertTrue(pacer.overloaded)
        self.assertEqual(pacer.read_limit(rate, 11.1), 0.0)  # not right away
        self.assertGreater(pacer.read_limit(rate, 12.1), 0.0)  # a second later


class TestDemoRecoversFromAHiccup(_WindowTestCase):
    demo = True

    def test_a_computer_at_six_tenths_of_real_time_recovers(self):
        clock = _FakeClock()
        rate = 2.4e6
        source = _FakeDemo(clock, cost=0.6 / rate)
        spoken = []
        self.win._audio.write = spoken.append
        self.win._audio_enabled = True
        self.win._control_panel.set_squelch_db(-100)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")

        def run(seconds):
            end = clock.now + seconds
            while clock.now < end:
                self.win._update_display()
                clock.advance(0.033)

        with mock.patch.object(self.mw, "time", clock):
            self.win._stop_acquisition()
            self.win._device = source  # still in demo mode
            self.win._start_acquisition()
            run(2.0)
            self.assertFalse(self.win._pacer.overloaded)
            self.assertTrue(spoken)
            source.cost = 2.0 / rate  # another program hogs the CPU
            run(1.5)
            self.assertTrue(self.win._pacer.overloaded)
            source.cost = 0.6 / rate  # over; this computer keeps up again
            run(4.0)
            self.assertFalse(self.win._pacer.overloaded)
            before = len(spoken)
            run(1.0)
            self.assertTrue(self.win._realtime())
            self.assertGreater(len(spoken), before)  # audio again
        self.win._stop_acquisition()
        self.win._device = None
        self.win._demo_mode = False
        self.win._refresh_state_ui()


if __name__ == "__main__":
    unittest.main()

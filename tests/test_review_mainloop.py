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

    def test_a_take_stopped_by_a_full_disk_keeps_that_message(self):
        self.win._toggle_recording(True)
        self.run_frames(3)
        self.dev._dropped_samples += 2 * TRANSFER
        # The disk fills up in the same frame as the drops.
        with mock.patch.object(
            self.win, "_check_recording_space", side_effect=self._disk_full
        ):
            self.run_frames(1)
        self.assertFalse(self.win._recording)
        self.assertIn("disk space", self.win._message_label.text())
        self.assertIn("13 ms lost", self.win._take_summary()[0])

    def _disk_full(self):
        self.win._stop_recording_with_warning("only 90 MB of disk space is left")
        return False


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


# ---------------------------------------------------------------------------
# audio-chain-balance-iq-zero-division
# ---------------------------------------------------------------------------


def _station(n, rate=2.4e6, seed=1):
    """Noise and a strong station 300 kHz above the center."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    x = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x += 0.5 * np.exp(2j * np.pi * 300e3 * t)
    return x.astype(np.complex64)


def _mirror_dbc(samples, rate=2.4e6):
    """The image at -300 kHz relative to the station at +300 kHz."""
    n = 65536
    spectrum = np.fft.fft(samples[:n] * np.hanning(n))
    k = int(round(300e3 / rate * n))
    return 20 * np.log10(abs(spectrum[-k]) / abs(spectrum[k]) + 1e-12)


class TestBalanceIqWithoutVariance(unittest.TestCase):
    RATE = 2.4e6

    def _conditioner(self):
        from sdr_module.gui.audio_chain import IqConditioner

        conditioner = IqConditioner()
        conditioner.balance_iq = True
        return conditioner

    def test_blocks_without_i_or_q_variance_pass_and_teach_nothing(self):
        rng = np.random.default_rng(3)
        for label, block in (
            ("zeros", np.zeros(80000, np.complex64)),
            ("Q = 0", (0.1 * rng.standard_normal(80000)).astype(np.complex64)),
            ("I = 0", (0.1j * rng.standard_normal(80000)).astype(np.complex64)),
            ("DC only", np.full(80000, 0.01 + 0.01j, np.complex64)),
        ):
            with self.subTest(block=label):
                conditioner = self._conditioner()
                for _ in range(3):
                    out = conditioner.process(block, self.RATE)
                    self.assertEqual(len(out), len(block))
                    self.assertTrue(np.all(np.isfinite(out)))
                self.assertEqual(conditioner.imbalance, (None, None))
                # The first real signal is corrected at once, not mirrored.
                out = conditioner.process(_station(80000), self.RATE)
                self.assertLess(_mirror_dbc(out), -40.0)
                gain, phase = conditioner.imbalance
                self.assertLess(abs(gain), 0.05)
                self.assertLess(abs(phase), 0.05)

    def test_a_stretch_of_zeros_does_not_drag_the_estimate(self):
        conditioner = self._conditioner()
        for _ in range(5):
            conditioner.process(_station(80000), self.RATE)
        settled = conditioner.imbalance
        for _ in range(60):  # two seconds of zeros
            conditioner.process(np.zeros(80000, np.complex64), self.RATE)
        self.assertEqual(conditioner.imbalance, settled)
        out = conditioner.process(_station(80000, seed=2), self.RATE)
        self.assertLess(_mirror_dbc(out), -40.0)

    def test_correction_refuses_a_degenerate_gain(self):
        from sdr_module.gui.audio_chain import correct_iq

        block = _station(1000)
        for gain in (-1.0, -0.9999999, float("nan"), float("-inf")):
            with self.subTest(gain=gain):
                out = correct_iq(block, gain, 0.0)
                self.assertTrue(np.all(np.isfinite(out)))
                self.assertLess(np.max(np.abs(out)), 10 * np.max(np.abs(block)))


class TestBalanceIqInTheWindow(_WindowTestCase):
    demo = True

    def test_zero_padding_in_a_recording_does_not_break_frames(self):
        self.win._iq_conditioner.balance_iq = True
        self.win._iq_conditioner.reset()
        seen = []
        self.win._spectrum.update_spectrum = seen.append
        for _ in range(3):
            self.win._process_block(np.zeros(80000, np.complex64))
        self.assertEqual(len(seen), 3)  # every frame plotted


# ---------------------------------------------------------------------------
# audio-sink-rtl-1msps-dropouts
# ---------------------------------------------------------------------------


class TestAudioSinkFeedsBetweenWrites(unittest.TestCase):
    RATE = 48000

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import audio_sink as audio_sink_module
        from tests.test_gui_audio_sink import Clock, FakeDevice, FakeOutput

        self.clock = Clock()
        patcher = mock.patch.object(audio_sink_module.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.FakeDevice, self.FakeOutput = FakeDevice, FakeOutput

    def make(self):
        from sdr_module.gui.audio_sink import AudioSink

        sink = AudioSink()
        output = self.FakeOutput(int(AudioSink.BUFFER_S * self.RATE) * 2)
        device = self.FakeDevice(output)
        sink._sink, sink._io, sink._sample_rate = output, device, self.RATE
        self.addCleanup(sink.stop)
        return sink, output

    @staticmethod
    def idle(sink):
        """The Qt event loop between display frames: the sink's feed timer
        (if it runs) gets its turn."""
        timer = getattr(sink, "_feed_timer", None)
        if timer is not None and timer.isActive():
            timer.timeout.emit()

    def play(self, sink, output, chunk_s, chunks):
        """``chunks`` chunks of ``chunk_s`` written every ``chunk_s``, the
        sound card playing 10 ms at a time; returns the milliseconds it ran
        dry once playing (an audible gap)."""
        dry_ms = 0.0
        started = False
        for _ in range(chunks):
            sink.write(np.full(int(round(chunk_s * self.RATE)), 0.25, np.float32))
            left = int(round(chunk_s * self.RATE))  # samples until the next
            while left > 0:
                step = min(left, int(0.01 * self.RATE))
                left -= step
                self.clock.now += step / self.RATE
                taken = output.play(2 * step)
                started = started or taken > 0
                if started:
                    dry_ms += 1000.0 * (2 * step - taken) / (2 * self.RATE)
                self.idle(sink)
        return dry_ms

    def test_a_chunk_longer_than_the_output_buffer_plays_without_gaps(self):
        sink, output = self.make()
        # An RTL-SDR at 1.0 MS/s: one 262 ms transfer's audio every 262 ms,
        # more than the output's 250 ms buffer.
        dry_ms = self.play(sink, output, 0.262, 40)
        self.assertLess(dry_ms, 15)  # (the first tick at most)
        self.assertEqual(sink.overruns, 0)

    def test_display_frames_still_play_as_before(self):
        sink, output = self.make()
        self.assertLess(self.play(sink, output, 0.033, 200), 15)
        self.assertEqual(sink.overruns, 0)
        self.assertEqual(sink.underruns, 0)

    def test_the_feed_stops_with_the_output(self):
        sink, output = self.make()
        sink.write(np.full(int(0.4 * self.RATE), 0.25, np.float32))
        self.assertTrue(sink._backlog)
        sink.stop()
        self.idle(sink)  # nothing left to feed, no error
        self.assertFalse(sink._backlog)
        timer = getattr(sink, "_feed_timer", None)
        self.assertTrue(timer is None or not timer.isActive())


# ---------------------------------------------------------------------------
# mainloop-rds-badge-survives-device-change
# ---------------------------------------------------------------------------


class TestRdsBadgeFollowsTheSource(_WindowTestCase):
    demo = True

    def tearDown(self):
        if HAS_PYQT6:
            self.win._is_running = False
            self.win._device = None
        super().tearDown()

    def _hear_station(self, frames=150):
        """The demo's station on 100.1 MHz until its name is known."""
        self.win._rds_auto_action.setChecked(True)
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        for _ in range(frames):  # (~5 s of the demo station)
            self.win._process_block(self.win._device.read_samples(80000))
            if self.win._rds_badge.text():
                break
        self.assertTrue(self.win._rds_badge.text())

    def _connect_noise_receiver(self):
        """Device > Connect... picks a receiver on the same frequency and
        sample rate that hears only noise."""
        from sdr_module.gui import device_dialog

        dev = _queued_hardware(self.win._device_sample_rate())
        dev.set_frequency(self.win._lo_frequency())

        class Dialog:
            def __init__(self, parent=None):
                pass

            def exec(self):
                return 1

            def deleteLater(self):
                pass

            def get_selected_device(self):
                return dev

        with mock.patch.object(device_dialog, "DeviceDialog", Dialog):
            self.win._show_device_dialog()
        self.assertIs(self.win._device, dev)
        self.assertAlmostEqual(self.win._current_frequency(), 100.1e6)
        return dev

    def _hear_noise(self, frames=10):
        rng = np.random.default_rng(4)
        for _ in range(frames):
            noise = 0.01 * (
                rng.standard_normal(80000) + 1j * rng.standard_normal(80000)
            )
            self.win._process_block(noise.astype(np.complex64))

    def test_a_new_receiver_on_the_same_frequency_forgets_the_station(self):
        self._hear_station()
        self._connect_noise_receiver()
        self.assertEqual(self.win._rds_badge.text(), "")
        self.assertFalse(self.win._rds_badge_action.isVisible())
        self.assertIsNone(self.win._rds_listener.station())
        # Its noise doesn't bring the old station's name back.
        self._hear_noise()
        self.assertEqual(self.win._rds_badge.text(), "")

    def test_stop_and_start_keep_the_station(self):
        self._hear_station()
        name = self.win._rds_badge.text()
        self.win._stop_acquisition()
        self.win._start_acquisition()
        self.assertEqual(self.win._rds_badge.text(), name)

    def test_the_decoder_tabs_rds_forgets_the_station_too(self):
        self.win._rds_auto_action.setChecked(True)
        self.win._decoder_panel._proto_combo.setCurrentText("RDS")
        self.settle(1)
        self._hear_station(frames=300)
        decoder = self.win._decoder
        self.assertTrue(getattr(decoder.station, "ps_name", ""))
        self._connect_noise_receiver()
        self.assertEqual(self.win._rds_badge.text(), "")
        self._hear_noise()
        self.assertEqual(self.win._rds_badge.text(), "")


# ---------------------------------------------------------------------------
# mainloop-stale-pre-nr-audio-replayed
# ---------------------------------------------------------------------------


class TestNoStaleAudio(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_an_empty_channel_block_leaves_no_audio_behind(self):
        mw = _mw()
        chain = mw._ReceiverChain(2.4e6, "FM", 25e3)
        rng = np.random.default_rng(5)
        block = (rng.standard_normal(80000) + 1j * rng.standard_normal(80000)).astype(
            np.complex64
        )
        audio = chain.demodulate(chain.channelize(block), 5e3)
        self.assertTrue(len(audio))
        self.assertTrue(len(chain.pre_nr_audio))
        chain.last_clip_fraction = 0.5
        audio = chain.demodulate(np.empty(0, np.complex64), 5e3)
        self.assertEqual(len(audio), 0)
        self.assertEqual(len(chain.pre_nr_audio), 0)
        self.assertEqual(chain.last_clip_fraction, 0.0)


class TestNoStaleAudioInTheWindow(_WindowTestCase):
    demo = True

    def test_sstv_and_the_tone_detector_get_each_block_once(self):
        self.win._stop_acquisition()
        self.win.set_frequency(145.8e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.win._sstv_panel._on_start_clicked()
        self.assertTrue(self.win._sstv_listening())
        heard, tones = [], []
        self.win._sstv_panel.process_audio = lambda audio, sample_rate=None: (
            heard.append(audio)
        )
        tone_process = self.win._tone_squelch.process
        self.win._tone_squelch.process = lambda audio: (
            tones.append(audio),
            tone_process(audio),
        )[1]
        dev = self.win._device
        dev.start_rx()
        # (25 kHz FM at 2.4 MS/s: 1:50 to the channel; 20 samples after an
        # odd-sized block give no channel sample.)
        for size in (80000, 80013, 20, 80000):
            self.win._process_block(dev.read_samples(size))
        given = [a for a in heard if len(a)]
        self.assertEqual(len({id(a) for a in given}), len(given))
        checked = [a for a in tones if len(a)]
        self.assertEqual(len({id(a) for a in checked}), len(checked))
        self.assertEqual(sum(len(a) for a in given), sum(len(a) for a in checked))


# ---------------------------------------------------------------------------
# mainloop-tone-state-depends-on-speaker
# ---------------------------------------------------------------------------


class TestToneStateWithoutTheSpeaker(_WindowTestCase):
    demo = True

    def _listen(self, frames=45):
        states = collections.Counter()
        present = collections.Counter()
        dev = self.win._device
        for _ in range(frames):
            self.win._process_block(dev.read_samples(80000))
            states[self.win._squelch_state(True)] += 1
            present[self.win._scanner_host.scan_signal_present()] += 1
        return states, present

    def setUp(self):
        super().setUp()
        self.win._stop_acquisition()
        self.win._device.start_rx()
        self.win._audio.write = lambda _audio: None
        # Demo NOAA weather radio: narrow FM, no CTCSS tone.
        self.win.set_frequency(162.55e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.win._control_panel.set_tone_squelch_hz(100.0)
        self.assertEqual(self.win._tone_squelch.tone_hz, 100.0)

    def test_a_carrier_without_the_tone_reads_tone_with_audio_off(self):
        self.win._set_audio_enabled(True)
        states, present = self._listen()
        self.assertEqual(states.most_common(1)[0][0], "tone")
        self.win._set_audio_enabled(False)
        states, present = self._listen()
        # The lamp (and the Info tab, and a scan) see the same: a signal,
        # but not the chosen tone; the speaker being off changes nothing.
        self.assertEqual(states.most_common(1)[0][0], "tone")
        self.assertEqual(present.most_common(1)[0][0], False)

    def test_with_the_tone_squelch_off_audio_off_reads_open(self):
        self.win._control_panel.set_tone_squelch_hz(0)
        self.win._set_audio_enabled(False)
        states, present = self._listen()
        self.assertEqual(states.most_common(1)[0][0], "open")
        self.assertEqual(present.most_common(1)[0][0], True)


if __name__ == "__main__":
    unittest.main()

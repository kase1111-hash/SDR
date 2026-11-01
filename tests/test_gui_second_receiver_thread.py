"""K-P2: the second receiver's heavy per-frame work runs on a worker thread.

Both windows share the GUI thread; the second receiver's channel filter,
demodulator and plot spectrum took a quarter of each frame of it. Now its
display loop (``_frame``) hands each frame's samples to a DSP worker
(``gui/frame_pipeline.FramePipeline``, one frame in flight) and presents
the last frame's result: the GUI thread reads, records, paints and plays;
no Qt object is touched by the worker; the audio is the same, in order;
what the worker keeps is never reset under it (a reset, a new FFT size or
sample rate, a retune and Stop wait for the frame in flight), and a frame
of another span is never painted on the new one.

``_process`` (one frame start to finish, as tests drive frames) keeps
working on the GUI thread.
"""

import threading
import unittest
from unittest import mock

import numpy as np

from tests.test_gui_chrome import HAS_PYQT6
from tests.test_gui_second_receiver import _FakeSink, _SecondCase

#: Samples a frame of these tests reads (fixed).
FRAME = 32768
#: How long a held DSP job waits on a build that runs it on the GUI thread.
HOLD_S = 5.0


def _on_gui_thread():
    return threading.current_thread() is threading.main_thread()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class _ThreadCase(_SecondCase):
    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()  # (every demo sample is read)
        self.addCleanup(patcher.stop)
        # The main window's audio is on (Listen decides who plays).
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None  # (as with an output device)
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()  # (its frames are run here)
        rx._pacer.demo_request = lambda rate, now: FRAME
        rx.device._rng = np.random.default_rng(5)

    def run_frames(self, count):
        for _ in range(count):
            self.rx._frame()

    def watch(self, obj, name, log, before=None):
        """Note the thread of each call of ``obj.name`` in ``log``."""
        real = getattr(obj, name)

        def call(*args, **kwargs):
            log.append(_on_gui_thread())
            if before is not None:
                before()
            return real(*args, **kwargs)

        setattr(obj, name, call)


class TestTheWorkerDoesTheHeavyWork(_ThreadCase):
    def test_filter_demodulator_and_plot_spectrum_run_off_the_gui_thread(self):
        rx = self.rx
        dsp, spectrum = [], []
        self.watch(rx._dsp, "process", dsp)
        self.watch(rx._frame_spectrum, "process", spectrum)
        self.run_frames(4)
        rx.stop()
        self.assertEqual(len(dsp), 4)
        self.assertEqual(len(spectrum), 4)
        self.assertNotIn(True, dsp)  # none on the GUI thread
        self.assertNotIn(True, spectrum)

    def test_a_frame_doesnt_wait_for_its_own_dsp(self):
        rx = self.rx
        gate, entered = threading.Event(), threading.Event()
        timeouts = []

        def hold():
            entered.set()
            if not gate.wait(HOLD_S):
                timeouts.append(1)

        self.watch(rx._dsp, "process", [], before=hold)
        self.run_frames(1)  # (returns while its DSP is held)
        self.assertTrue(entered.wait(HOLD_S))
        self.assertEqual(timeouts, [])
        gate.set()
        rx.stop()
        self.assertEqual(timeouts, [])

    def test_qt_is_touched_only_on_the_gui_thread(self):
        rx = self.rx
        sink = self.sink()
        rx.set_listen("second")
        rx.tune(145.29e6, recenter=True)  # (the demo repeater: audio)
        rx._sync_audio_output()
        calls = []
        self.watch(rx._spectrum, "update_spectrum", calls)
        self.watch(rx._waterfall, "add_line", calls)
        self.watch(rx._level_meter, "set_reading", calls)
        self.watch(sink, "write", calls)
        self.run_frames(5)
        rx.stop()
        self.assertGreaterEqual(len(calls), 15)
        self.assertNotIn(False, calls)


class TestTheSameAudioInOrder(_ThreadCase):
    def test_audio_matches_the_frames_one_by_one(self):
        from sdr_module.gui.second_receiver import ReceiverDsp

        rx = self.rx
        sink = self.sink()
        rx.set_listen("second")
        rx.tune(100.1e6, recenter=True)
        rx.set_mode("FM", "75 kHz", "200 kHz")
        rx._sync_audio_output()
        self.assertTrue(sink.open)
        blocks = []
        real = rx._dsp.process

        def keep(samples, play):
            blocks.append((np.array(samples), play))
            return real(samples, play)

        rx._dsp.process = keep
        # The same settings, the same blocks, one at a time on this thread.
        ref = ReceiverDsp()
        for name in ("rate", "mode", "bandwidth_hz", "deviation_hz", "offset_hz"):
            setattr(ref, name, getattr(rx._dsp, name))
        ref.options = rx._dsp.options
        gate = rx._dsp.squelch
        ref.squelch.enabled, ref.squelch.auto = gate.enabled, gate.auto
        ref.squelch.manual_db = gate.manual_db
        self.run_frames(12)
        rx.stop()  # (the last frame's audio is played before it fades out)
        self.assertEqual(len(blocks), 12)
        expected = []
        for samples, play in blocks:
            audio = ref.process(samples, play).audio
            if audio is not None and len(audio):
                expected.append(audio)
        expected = np.concatenate(expected)
        # (12 frames of 13.7 ms at 48 kHz: 7864 samples, the squelch open.)
        self.assertGreater(len(expected), 7000)
        played = sink.samples()
        np.testing.assert_allclose(played[: len(expected)], expected, atol=1e-6)


class TestNothingChangesUnderTheWorker(_ThreadCase):
    def hold_next_job(self, obj, name):
        """Hold ``obj.name`` (run by the worker) until released; returns
        (release, running): ``running`` is set while it runs."""
        gate, running = threading.Event(), threading.Event()
        real = getattr(obj, name)

        def held(*args, **kwargs):
            running.set()
            try:
                gate.wait(HOLD_S)
                return real(*args, **kwargs)
            finally:
                running.clear()

        setattr(obj, name, held)
        self.addCleanup(gate.set)
        return gate, running

    def overlaps(self, obj, name, running):
        """Note whether each call of ``obj.name`` came while ``running``."""
        seen = []
        real = getattr(obj, name)

        def call(*args, **kwargs):
            seen.append(running.is_set())
            return real(*args, **kwargs)

        setattr(obj, name, call)
        return seen

    def test_stop_waits_for_the_frame_in_flight(self):
        rx = self.rx
        gate, running = self.hold_next_job(rx._dsp, "process")
        resets = self.overlaps(rx._dsp, "reset", running)
        self.run_frames(1)
        threading.Timer(0.05, gate.set).start()
        rx.stop()
        self.assertTrue(resets)
        self.assertNotIn(True, resets)

    def test_a_new_fft_size_waits_for_the_frame_in_flight(self):
        rx = self.rx
        gate, running = self.hold_next_job(rx._frame_spectrum, "process")
        configures = self.overlaps(rx._frame_spectrum, "configure", running)
        self.run_frames(1)
        threading.Timer(0.05, gate.set).start()
        rx._apply_plot_resolution(8192, "hann")
        self.assertEqual(configures, [False])
        self.assertEqual(rx._frame_spectrum.size, 8192)
        self.run_frames(2)
        rx.stop()

    def test_a_new_sample_rate_waits_for_the_frame_in_flight(self):
        rx = self.rx
        gate, running = self.hold_next_job(rx._dsp, "process")
        resets = self.overlaps(rx._dsp, "reset", running)
        self.run_frames(1)
        threading.Timer(0.05, gate.set).start()
        index = rx._rate_combo.findText("3.2 MS/s")
        self.assertGreaterEqual(index, 0)
        rx._rate_combo.setCurrentIndex(index)
        rx._rate_combo.activated.emit(index)
        self.assertEqual(rx._rate(), 3.2e6)
        self.assertTrue(resets)
        self.assertNotIn(True, resets)

    def test_a_retune_paints_the_frame_in_flight_on_its_own_span(self):
        rx = self.rx
        gate, running = self.hold_next_job(rx._frame_spectrum, "process")
        painted = []
        real = rx._spectrum.update_spectrum

        def update(*args, **kwargs):
            painted.append(float(rx._lo_hz))
            return real(*args, **kwargs)

        rx._spectrum.update_spectrum = update
        self.run_frames(1)
        self.assertTrue(running.wait(HOLD_S))
        lo = float(rx._lo_hz)
        threading.Timer(0.05, gate.set).start()
        self.assertTrue(rx.tune(lo + 2 * rx._rate(), recenter=True))
        self.assertNotEqual(float(rx._lo_hz), lo)  # (a new span)
        self.assertEqual(painted, [lo])  # (painted before the LO moved)
        self.assertFalse(rx._pipeline.pending)
        rx.stop()

    def test_a_frame_of_another_span_is_not_painted(self):
        rx = self.rx
        painted, readings = [], []
        self.watch(rx._spectrum, "update_spectrum", painted)
        self.watch(rx._level_meter, "set_reading", readings)
        lo, rate = float(rx._lo_hz), float(rx._rate())
        samples = rx.device.read_samples(FRAME)
        rx._present(rx._work(samples, False, True, (lo - 1e6, rate)))
        self.assertEqual(painted, [])  # (its spectrum belongs elsewhere)
        self.assertEqual(len(readings), 1)  # (its level still shows)
        rx._present(rx._work(samples, False, True, (lo, rate)))
        self.assertEqual(len(painted), 1)
        rx.stop()


class TestWhatItStillDoes(_ThreadCase):
    def test_a_hidden_window_skips_its_plots_not_its_receiver(self):
        rx = self.rx
        dsp, spectrum = [], []
        self.watch(rx._dsp, "process", dsp)
        self.watch(rx._frame_spectrum, "process", spectrum)
        rx.showMinimized()
        self.settle()
        self.run_frames(3)
        rx.showNormal()
        self.settle()
        self.run_frames(2)
        rx.stop()
        self.assertEqual(len(dsp), 5)
        self.assertEqual(len(spectrum), 2)

    def test_a_recording_keeps_every_frame_in_order(self):
        import os
        import shutil
        import tempfile

        rx = self.rx
        folder = tempfile.mkdtemp(prefix="rx2-thread-rec-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        fed = []
        real = rx._dsp.process

        def keep(samples, play):
            fed.append(np.array(samples, np.complex64))
            return real(samples, play)

        rx._dsp.process = keep
        self.assertTrue(rx._set_recording(True))
        self.run_frames(5)
        rx._set_recording(False)
        rx.stop()
        names = [n for n in os.listdir(folder) if n.endswith(".sigmf-data")]
        self.assertEqual(len(names), 1, names)
        data = np.fromfile(os.path.join(folder, names[0]), np.complex64)
        np.testing.assert_array_equal(data, np.concatenate(fed))

    def test_every_frame_reaches_the_dsp_once_in_order(self):
        # (One frame in flight at a time: none piles up, none is dropped.)
        rx = self.rx
        fed, heard = [], []
        real_feed, real_process = rx._feed, rx._dsp.process

        def feed(samples):
            fed.append(complex(samples[0]))
            return real_feed(samples)

        def process(samples, play):
            heard.append(complex(samples[0]))
            return real_process(samples, play)

        rx._feed = feed
        rx._dsp.process = process
        for _ in range(10):
            rx._frame()
            self.assertTrue(rx._pipeline.pending)  # (this frame's, in flight)
        rx.stop()
        self.assertFalse(rx._pipeline.pending)
        self.assertEqual(len(fed), 10)
        self.assertEqual(heard, fed)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDirectFramesStayOnThisThread(_SecondCase):
    def test_process_is_one_frame_start_to_finish(self):
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        sink = _FakeSink()
        rx._audio = sink
        calls = []
        real = rx._dsp.process
        rx._dsp.process = lambda s, play: calls.append(_on_gui_thread()) or real(
            s, play
        )
        rx._process(rx.device.read_samples(80_000))
        self.assertEqual(calls, [True])
        self.assertTrue(rx._spectrum._has_data)  # (painted already)
        rx.stop()

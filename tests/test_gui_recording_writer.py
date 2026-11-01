"""K-P1: an I/Q recording's writes go to a writer thread.

At 20 MS/s a take writes 160 MB a second. Written from the display loop,
each frame's write took 20-25 ms of the GUI thread's 33 ms (over 100 ms
while the disk caught up): frames fell behind, the window counted itself
overloaded and audio paused. Now the GUI thread only queues a copy of each
block for the take's writer thread, a bounded queue: the take still keeps
every sample in order; a block the queue can't hold (the disk can't keep
up) is counted as lost and said, as samples a driver drops are; a write
that fails stops the recording with the same message as before; Stop writes
everything queued before the take is offered for saving; the free-space
check counts what is still queued. The second receiver records through the
same writer.

A fake disk stands in for the take's file (``os.fdopen`` as the recording
buffer opens it): it notes the thread of each write and can hold writes up
(``gate``) or run out of room. On a build that writes on the GUI thread a
held write gives up after ``HOLD_S`` (counted in ``timeouts``), so these
tests fail there instead of hanging.
"""

import errno
import os
import threading
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_recording_files import _FilesCase  # noqa: E402

#: Samples per display frame (fixed, so the tests know what each one adds).
FRAME = 16384
#: How long a held write waits on a build that writes on the GUI thread.
HOLD_S = 5.0


class _Disk:
    """The take's file system: counts writes (and those on the GUI thread),
    holds them while ``gate`` is clear, and has ``room`` bytes left (None:
    plenty), after which a write fails as on a full disk."""

    def __init__(self):
        self.writes = 0
        self.on_gui_thread = 0
        self.gate = threading.Event()
        self.gate.set()
        self.waiting = threading.Event()  # a write waits at the gate
        self.timeouts = 0
        self.room = None
        self.failed = threading.Event()
        self.written = 0

    def write(self, handle, data):
        self.writes += 1
        if threading.current_thread() is threading.main_thread():
            self.on_gui_thread += 1
        if not self.gate.is_set():
            self.waiting.set()
            if not self.gate.wait(HOLD_S):
                self.timeouts += 1  # (written from the GUI thread: let it go)
                self.gate.set()
        size = len(memoryview(data))
        if self.room is not None and self.written + size > self.room:
            left = self.room - self.written
            if left <= 0:
                self.failed.set()
                raise OSError(errno.ENOSPC, "No space left on device")
            data = memoryview(data)[:left]  # a short write, as on a full disk
        wrote = handle.write(data)
        self.written += wrote
        return wrote


class _HeldFile:
    """The take's file, written through the fake disk."""

    def __init__(self, handle, disk):
        self._handle = handle
        self._disk = disk

    def write(self, data):
        return self._disk.write(self._handle, data)

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _settle_writes(win):
    """Wait until the take's writer has written what was queued (a build
    without one wrote it in the frame already)."""
    flush = getattr(win._samples_buffer, "flush", None)
    if callable(flush):
        flush()


def _writer_threads():
    return [t for t in threading.enumerate() if "recording writer" in t.name]


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class _WriterCase(_FilesCase):
    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        self.disk = _Disk()
        real = os.fdopen

        def fdopen(fd, *args, **kwargs):
            handle = real(fd, *args, **kwargs)
            if args[:1] == ("wb",) and kwargs.get("buffering") == 0:
                return _HeldFile(handle, self.disk)
            return handle

        for patcher in (
            mock.patch.object(self.mw.os, "fdopen", fdopen),
            # Every demo sample is read: the demo doesn't skip ahead to the
            # clock (a test held up at the gate would see samples lost).
            mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.disk.gate.set)  # never leave a write waiting
        self.frame_size(FRAME)
        # The blocks the take is given, as the frames give them.
        self.captured = []
        win = self.win
        capture = win._capture

        def keep(samples):
            if win._recording and not win._recording_paused:
                self.captured.append(np.array(samples, np.complex64))
            return capture(samples)

        win._capture = keep

    def frame_size(self, count):
        self.win._pacer.demo_request = lambda rate, now: count

    def start(self):
        win = self.win
        win._toggle_recording(True)
        win._sync_record_buttons()
        self.assertTrue(win._recording)

    def stop(self):
        win = self.win
        win._toggle_recording(False)
        win._sync_record_buttons()

    def frames(self, count):
        for _ in range(count):
            self.win._update_display()

    def take(self):
        return np.asarray(self.win._samples_buffer.samples())


class TestTheGuiThreadOnlyQueues(_WriterCase):
    def test_the_writer_thread_writes_every_sample_in_order(self):
        self.start()
        self.frames(4)
        self.stop()
        self.assertGreater(self.disk.writes, 0)
        self.assertEqual(self.disk.on_gui_thread, 0)  # (all four frames' writes)
        expected = np.concatenate(self.captured)
        self.assertEqual(len(expected), 4 * FRAME)
        np.testing.assert_array_equal(self.take(), expected)
        self.assertEqual(os.path.getsize(self.win._samples_buffer.path), 4 * FRAME * 8)

    def test_a_slow_disk_doesnt_hold_up_the_frames(self):
        self.start()
        self.disk.gate.clear()  # the disk takes nothing for now
        self.frames(4)
        # Every frame went on while the first write waited at the gate.
        self.assertTrue(self.disk.waiting.wait(HOLD_S))
        self.assertEqual(self.disk.timeouts, 0)
        self.assertEqual(self.win._buffer_sample_count(), 4 * FRAME)  # REC size
        self.assertEqual(self.win._recording_bytes, 4 * FRAME * 8)
        self.disk.gate.set()
        self.stop()
        self.assertEqual(self.disk.timeouts, 0)
        np.testing.assert_array_equal(self.take(), np.concatenate(self.captured))

    def test_the_writer_gets_a_copy_the_frame_no_longer_touches(self):
        win = self.win
        self.start()
        self.frames(1)
        self.disk.gate.clear()
        block = (np.arange(FRAME) * (1 + 1j) / FRAME).astype(np.complex64)
        original = block.copy()
        win._capture(block)
        block[:] = 0  # the caller's array, reused
        self.disk.gate.set()
        self.stop()
        np.testing.assert_array_equal(self.take()[-FRAME:], original)


class TestNothingIsLostSilently(_WriterCase):
    def test_a_block_the_queue_cant_hold_is_counted_as_lost_and_said(self):
        win = self.win
        # A queue that holds one frame's block: the next ones wait for room.
        with mock.patch.object(self.mw, "_CAPTURE_QUEUE_S", 1e-6, create=True):
            self.start()
            self.disk.gate.clear()
            self.frames(4)
            self.assertEqual(self.disk.timeouts, 0)
            self.assertEqual(win._buffer_sample_count(), FRAME)  # (the first)
            self.assertEqual(win._take_lost[1], 3 * FRAME)
            text = self.message()
            self.assertIn("The recording has gaps:", text)
            self.assertIn("couldn't keep up", text)
            self.disk.gate.set()
            self.stop()
        np.testing.assert_array_equal(self.take(), self.captured[0])
        summary = win._take_summary()[0]
        self.assertIn("lost", summary)

    def test_a_failed_write_stops_the_recording_as_before(self):
        win = self.win
        self.start()
        self.frames(1)
        _settle_writes(win)
        self.disk.room = int(2.5 * FRAME * 8) + 12  # (not a whole sample)
        for _ in range(10):
            self.frames(1)
            _settle_writes(win)
            if not win._recording:
                break
        else:
            self.frames(1)
        self.assertFalse(win._recording)
        store = win._samples_buffer
        count = store.sample_count
        # Every sample counted is on disk, nothing beyond it.
        self.assertEqual(count, self.disk.room // 8)
        self.assertEqual(os.path.getsize(store.path), count * 8)
        text = self.message()
        self.assertTrue(
            text.startswith("Recording stopped: the recording file can't be written ("),
            text,
        )
        self.assertIn("No space left on device", text)
        self.assertIn("on another drive if this one is full", text)
        np.testing.assert_array_equal(
            self.take(), np.concatenate(self.captured)[:count]
        )

    def test_a_write_failing_after_the_last_frame_is_still_reported(self):
        win = self.win
        self.start()
        self.frames(1)
        _settle_writes(win)
        self.disk.room = int(1.5 * FRAME * 8)
        self.frames(1)  # its write fails (here, or on the writer thread)
        self.assertTrue(self.disk.failed.wait(HOLD_S))
        _settle_writes(win)
        win._on_panel_record_paused(True)  # no frame captures from now on
        win._update_status()  # (the status timer)
        self.assertFalse(win._recording)
        self.assertIn("the recording file can't be written", self.message())
        store = win._samples_buffer
        self.assertEqual(os.path.getsize(store.path), store.sample_count * 8)

    def test_a_write_failing_as_a_block_is_queued_is_said_once(self):
        # The writer fails after the frame looked for a failed write and
        # before it queued its own block: the recording stops, said once,
        # and the take is what the file holds.
        win = self.win
        self.start()
        self.frames(1)
        _settle_writes(win)
        store = win._samples_buffer
        self.disk.room = int(1.5 * FRAME * 8)
        self.disk.gate.clear()
        self.frames(1)  # (its write waits at the gate, then fails)
        self.assertTrue(self.disk.waiting.wait(HOLD_S))
        real = store._raise_failure

        def fail_meanwhile():
            if not self.disk.failed.is_set():
                self.disk.gate.set()
                self.assertTrue(self.disk.failed.wait(HOLD_S))
                store._writer.flush()
            return real()

        store._raise_failure = fail_meanwhile
        with self.assertLogs("sdr_module.gui.main_window", level="WARNING") as logs:
            self.frames(1)
        said = [line for line in logs.output if "Recording stopped" in line]
        self.assertEqual(len(said), 1, said)
        self.assertFalse(win._recording)
        self.assertEqual(store.sample_count, int(1.5 * FRAME))
        self.assertEqual(os.path.getsize(store.path), store.sample_count * 8)

    def test_a_write_failing_as_stop_writes_the_queue_is_said(self):
        # The disk fills up while Stop writes out what the writer still
        # holds: the recording stops saying why, as in a frame, and the take
        # is what the file holds (not "Recorded ... samples").
        win = self.win
        self.start()
        self.frames(1)
        _settle_writes(win)
        self.disk.room = int(1.5 * FRAME * 8)
        self.disk.gate.clear()
        self.frames(1)  # (its write waits at the gate, then fails)
        threading.Timer(0.05, self.disk.gate.set).start()
        self.stop()
        self.assertFalse(win._recording)
        text = self.message()
        self.assertTrue(
            text.startswith("Recording stopped: the recording file can't be written ("),
            text,
        )
        store = win._samples_buffer
        self.assertEqual(store.sample_count, int(1.5 * FRAME))
        self.assertEqual(os.path.getsize(store.path), store.sample_count * 8)
        self.assertEqual(self.disk.timeouts, 0)


class TestStopWritesEverythingFirst(_WriterCase):
    def test_the_take_is_whole_when_it_is_offered_for_saving(self):
        win = self.win
        win._control_panel._auto_save_check.setChecked(True)
        self.start()
        self.disk.gate.clear()
        self.frames(3)
        threading.Timer(0.05, self.disk.gate.set).start()
        self.stop()  # waits for the writer, then saves (moves the file)
        self.assertEqual(self.disk.timeouts, 0)
        self.assertFalse(win._buffer_unsaved)
        saved = win._last_saved
        self.assertTrue(saved and os.path.exists(saved), saved)
        self.assertEqual(os.path.getsize(saved), 3 * FRAME * 8)
        np.testing.assert_array_equal(
            np.fromfile(saved, np.complex64), np.concatenate(self.captured)
        )
        self.assertEqual(_writer_threads(), [])

    def test_closing_the_window_while_recording_leaves_no_writer(self):
        self.start()
        self.frames(2)
        self.win._buffer_unsaved = False  # (nothing to ask)
        self.win.close()
        self.settle()
        self.assertEqual(_writer_threads(), [])


class TestFreeSpaceCountsWhatIsQueued(_WriterCase):
    def test_the_space_check_counts_samples_not_yet_on_disk(self):
        from collections import namedtuple

        win = self.win
        big = 1 << 17  # 1 MB a frame
        self.frame_size(big)
        usage = namedtuple("Usage", "total used free")(
            10**12, 0, self.mw._MIN_FREE_BYTES + 1_500_000
        )
        with mock.patch.object(self.mw.shutil, "disk_usage", return_value=usage):
            self.start()
            self.disk.gate.clear()
            self.frames(2)  # 2 MB queued, none of it on disk yet
            self.assertTrue(win._recording)
            win._next_space_check = 0.0
            threading.Timer(0.2, self.disk.gate.set).start()
            self.frames(1)
        self.assertFalse(win._recording)
        self.assertIn("of disk space is left", self.message())
        self.assertEqual(self.disk.timeouts, 0)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSecondReceiverRecordsThroughTheWriter(_FilesCase):
    def setUp(self):
        super().setUp()
        from sdr_module.dsp.recording import IQRecorder
        from sdr_module.gui import device_dialog

        self.store.values["recording/folder"] = self.folder
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()
        self.timeouts = 0
        self.fail = None
        real = IQRecorder.write
        test = self

        def write(recorder, samples):
            test.calls.append(threading.current_thread() is threading.main_thread())
            if not test.gate.wait(HOLD_S):
                test.timeouts += 1
                test.gate.set()
            if test.fail is not None:
                raise test.fail
            return real(recorder, samples)

        for patcher in (
            mock.patch.object(IQRecorder, "write", write),
            mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.gate.set)
        self.rx = self.win._second_rx.show()
        self.rx._status_timer.stop()
        self.rx._pacer.demo_request = lambda rate, now: FRAME
        self.addCleanup(self.rx.shutdown)
        self.assertTrue(self.rx.start())
        self.rx.tune(145.8e6, recenter=True)

    def frames(self, count):
        for _ in range(count):
            self.rx._frame()

    def data_file(self):
        names = [n for n in os.listdir(self.folder) if n.endswith(".sigmf-data")]
        self.assertEqual(len(names), 1, names)
        return os.path.join(self.folder, names[0])

    def test_its_writes_happen_on_the_writer_thread(self):
        rx = self.rx
        self.assertTrue(rx._set_recording(True))
        self.gate.clear()
        self.frames(4)
        self.assertEqual(self.timeouts, 0)
        self.gate.set()
        rx._set_recording(False)
        self.assertTrue(self.calls)
        self.assertNotIn(True, self.calls)  # none on the GUI thread
        self.assertEqual(os.path.getsize(self.data_file()), 4 * FRAME * 8)
        self.assertTrue(rx.message().startswith("Saved iq_"), rx.message())
        self.assertEqual(_writer_threads(), [])

    def test_a_block_the_queue_cant_hold_is_counted_and_said(self):
        rx = self.rx
        with mock.patch.object(self.mw, "_CAPTURE_QUEUE_S", 1e-6, create=True):
            self.assertTrue(rx._set_recording(True))
            self.gate.clear()
            self.frames(4)
            self.assertEqual(self.timeouts, 0)
            self.assertIn("The recording has gaps:", rx.message())
            self.gate.set()
            rx._set_recording(False)
        text = rx.message()
        self.assertTrue(text.startswith("Saved iq_"), text)
        self.assertIn("lost", text)
        self.assertEqual(os.path.getsize(self.data_file()), FRAME * 8)

    def test_a_failed_write_stops_the_recording(self):
        rx = self.rx
        self.assertTrue(rx._set_recording(True))
        self.frames(1)
        self.fail = OSError(errno.ENOSPC, "No space left on device")
        for _ in range(10):
            self.frames(1)
            writer = getattr(rx, "_record_writer", None)
            if writer is not None:
                writer.flush()
            if rx._recorder is None:
                break
        else:
            self.frames(1)
        self.assertIsNone(rx._recorder)
        text = rx.message()
        self.assertTrue(
            text.startswith("Recording stopped: the file can't be written ("), text
        )
        self.assertIn("No space left on device", text)
        self.assertFalse(rx._record_button.isChecked())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheTakesStore(unittest.TestCase):
    """``_SampleStore.queue``: the take's blocks through its writer thread."""

    def setUp(self):
        import shutil
        import tempfile

        from sdr_module.gui import main_window

        self.mw = main_window
        self.folder = tempfile.mkdtemp(prefix="sdr-store-writer-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.disk = _Disk()
        self.addCleanup(self.disk.gate.set)
        real = os.fdopen

        def fdopen(fd, *args, **kwargs):
            handle = real(fd, *args, **kwargs)
            if args[:1] == ("wb",) and kwargs.get("buffering") == 0:
                return _HeldFile(handle, self.disk)
            return handle

        patcher = mock.patch.object(self.mw.os, "fdopen", fdopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def store(self, queue_bytes=1 << 24):
        store = self.mw._SampleStore(self.folder, queue_bytes=queue_bytes)
        self.addCleanup(store.discard)
        return store

    @staticmethod
    def blocks(count=4, n=1000):
        return [
            (np.arange(i * n, (i + 1) * n) * (1 + 1j)).astype(np.complex64)
            for i in range(count)
        ]

    def test_blocks_read_back_as_queued(self):
        store = self.store()
        blocks = self.blocks()
        for block in blocks:
            self.assertTrue(store.queue(block))
        self.assertEqual(len(store), 4)  # counted as queued (the REC size)
        self.assertEqual(store.sample_count, 4000)
        self.assertEqual(store, blocks)  # (reading waits for the writer)
        store.finish()
        self.assertEqual(os.path.getsize(store.path), 4000 * 8)
        self.assertEqual(self.disk.on_gui_thread, 0)
        self.assertEqual(_writer_threads(), [])

    def test_other_sample_types_are_written_as_complex64(self):
        store = self.store()
        block = np.arange(10) * (0.5 - 0.25j)  # complex128
        self.assertTrue(store.queue(block))
        np.testing.assert_array_equal(store.samples(), block.astype(np.complex64))

    def test_a_failed_write_keeps_what_the_file_holds_and_is_raised_once(self):
        store = self.store()
        blocks = self.blocks()
        self.disk.room = int(1.5 * 1000 * 8) + 4  # (half a sample over)
        self.disk.gate.clear()  # (all four queued before any is written)
        for block in blocks:
            self.assertTrue(store.queue(block))
        self.disk.gate.set()
        store.flush()
        with self.assertRaises(OSError) as caught:
            store.check()
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        store.check()  # (once)
        store.finish()  # (nothing left to say)
        # Block 0, and the first half of block 1 as a block of its own.
        self.assertEqual(store.sample_count, 1500)
        self.assertEqual(len(store), 2)
        np.testing.assert_array_equal(store[0], blocks[0])
        np.testing.assert_array_equal(store[1], blocks[1][:500])
        self.assertEqual(os.path.getsize(store.path), 1500 * 8)

    def test_a_write_failing_as_a_block_is_queued_is_raised_once(self):
        # (It failed after queue() looked for a failure, before its put.)
        store = self.store()
        blocks = self.blocks()
        self.disk.room = int(1.5 * 1000 * 8)
        self.assertTrue(store.queue(blocks[0]))
        store.flush()
        self.disk.gate.clear()
        self.assertTrue(store.queue(blocks[1]))  # (waits at the gate, fails)
        self.assertTrue(self.disk.waiting.wait(HOLD_S))
        real = store._raise_failure

        def fail_meanwhile():
            if not self.disk.failed.is_set():
                self.disk.gate.set()
                self.assertTrue(self.disk.failed.wait(HOLD_S))
                store._writer.flush()
            return real()

        store._raise_failure = fail_meanwhile
        with self.assertRaises(OSError):
            store.queue(blocks[2])
        store.check()  # (raised once: nothing more to say)
        store.finish()
        self.assertEqual(store.sample_count, 1500)
        self.assertEqual(os.path.getsize(store.path), 1500 * 8)

    def test_an_imported_recording_takes_no_more(self):
        store = self.mw._SampleStore.from_samples(np.ones(8, np.complex64))
        with self.assertRaises(ValueError):
            store.queue(np.ones(8, np.complex64))

    def test_discard_ends_the_writer_and_deletes_the_file(self):
        store = self.store()
        self.disk.gate.clear()
        store.queue(self.blocks(1)[0])
        store.queue(self.blocks(1)[0])
        self.assertTrue(self.disk.waiting.wait(HOLD_S))
        path = store.path
        threading.Timer(0.05, self.disk.gate.set).start()
        store.discard()  # (the write under way ends; the other is dropped)
        self.assertEqual(_writer_threads(), [])
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.disk.timeouts, 0)

    def test_a_store_dropped_while_writing_leaves_nothing_behind(self):
        import gc

        store = self.mw._SampleStore(self.folder, queue_bytes=1 << 24)
        store.queue(self.blocks(1)[0])
        path = store.path
        del store
        gc.collect()
        self.assertEqual(_writer_threads(), [])
        self.assertFalse(os.path.exists(path))


class TestAStretchAfterAPause(_WriterCase):
    def test_starts_with_the_first_block_kept(self):
        win = self.win
        first, second = "2026-11-01T10:00:00.000Z", "2026-11-01T10:00:01.000Z"
        stamps = iter([first, second])
        with mock.patch.object(self.mw, "_CAPTURE_QUEUE_S", 1e-6, create=True):
            self.start()
            self.disk.gate.clear()
            self.frames(1)  # (held at the gate: the queue is full)
            win._on_panel_record_paused(True)
            self.frames(1)  # (paused: not captured)
            win._on_panel_record_paused(False)
            with mock.patch.object(self.mw, "_utc_stamp", lambda: next(stamps)):
                self.frames(1)  # the first block after the pause: refused
                self.assertEqual(win._take_lost[1], FRAME)
                self.disk.gate.set()
                _settle_writes(win)
                self.frames(1)  # the first block kept after the pause
            self.stop()
        # (The stretch after the pause starts with its first block kept,
        # and when that was received.)
        self.assertEqual(win._samples_buffer.segments, [(FRAME, second)])

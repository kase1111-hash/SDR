"""K-P1: ``dsp/background_writer.BackgroundWriter`` writes blocks of samples
in order on a thread of its own, so the display loop never waits for the
disk (an I/Q recording at 20 MS/s wrote 160 MB a second from the GUI
thread). Its queue is bounded: a block that doesn't fit is refused, for the
caller to count as lost and say so (never lost silently), and the first
error a write raises stops the writer and is kept for the caller to report
on its own thread.
"""

import threading

import numpy as np
import pytest

from sdr_module.dsp import background_writer
from sdr_module.dsp.background_writer import BackgroundWriter


class _Disk:
    """A write function standing in for a file: keeps each block written
    and the threads that wrote; while ``gate`` is clear a write waits (a
    disk catching up); from block ``fail_at`` on a write raises."""

    def __init__(self):
        self.blocks = []
        self.threads = set()
        self.gate = threading.Event()
        self.gate.set()
        self.waiting = threading.Event()  # a write is waiting at the gate
        self.fail_at = None
        self.error = OSError(28, "No space left on device")

    def __call__(self, block):
        self.threads.add(threading.get_ident())
        if not self.gate.is_set():
            self.waiting.set()
        assert self.gate.wait(10), "the test never let the write through"
        if self.fail_at is not None and len(self.blocks) >= self.fail_at:
            raise self.error
        self.blocks.append(np.array(block))


def _block(value, n=100):
    return np.full(n, value, np.complex64)


def _values(blocks):
    return [int(block[0].real) for block in blocks]


def test_blocks_are_written_in_order_on_a_thread_of_its_own():
    disk = _Disk()
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    blocks = [np.arange(i * 100, (i + 1) * 100).astype(np.complex64) for i in range(20)]
    for block in blocks:
        assert writer.put(block)
    assert writer.close()
    assert disk.threads and threading.get_ident() not in disk.threads
    np.testing.assert_array_equal(np.concatenate(disk.blocks), np.concatenate(blocks))
    assert writer.written == 2000
    assert writer.pending_bytes == 0
    assert writer.error is None
    assert not writer.running  # (close ends the thread)


def test_the_caller_never_waits_for_the_disk():
    disk = _Disk()
    disk.gate.clear()  # the disk takes nothing for now
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    for value in range(5):
        assert writer.put(_block(value))  # returns at once, every time
    assert disk.waiting.wait(5)
    assert disk.blocks == []
    assert writer.pending_bytes == 5 * 800
    disk.gate.set()
    assert writer.flush()
    assert _values(disk.blocks) == [0, 1, 2, 3, 4]
    assert writer.pending_bytes == 0
    writer.close()


def test_put_hands_over_a_copy():
    disk = _Disk()
    disk.gate.clear()
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    block = _block(1)
    assert writer.put(block)
    block[:] = 0  # the caller reuses its array straight away
    disk.gate.set()
    writer.close()
    np.testing.assert_array_equal(disk.blocks[0], _block(1))


def test_a_full_queue_refuses_a_block_and_keeps_the_others_in_order():
    disk = _Disk()
    disk.gate.clear()
    writer = BackgroundWriter(disk, max_bytes=3 * 800)
    taken = [writer.put(_block(value)) for value in range(5)]
    assert taken == [True, True, True, False, False]
    assert writer.pending_bytes == 3 * 800
    disk.gate.set()
    assert writer.flush()
    assert writer.put(_block(9))  # room again
    writer.close()
    assert _values(disk.blocks) == [0, 1, 2, 9]
    assert writer.written == 400


def test_a_block_bigger_than_the_queue_goes_when_nothing_waits():
    disk = _Disk()
    writer = BackgroundWriter(disk, max_bytes=800)
    assert writer.put(_block(1, n=1000))  # (a catch-up frame: never refused alone)
    writer.close()
    assert writer.written == 1000


def test_an_empty_block_is_nothing_to_write():
    disk = _Disk()
    writer = BackgroundWriter(disk, max_bytes=800)
    assert writer.put(np.zeros(0, np.complex64))
    assert not writer.running
    assert writer.close()
    assert disk.blocks == []


def test_the_first_error_stops_the_writer_and_waits_for_the_caller():
    disk = _Disk()
    disk.gate.clear()
    disk.fail_at = 2
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    for value in range(5):
        assert writer.put(_block(value))
    disk.gate.set()
    assert writer.flush()  # (done: nothing more will be written)
    assert writer.error is disk.error
    assert _values(disk.blocks) == [0, 1]
    assert writer.written == 200
    assert writer.pending_bytes == 0  # what was queued after it is dropped
    with pytest.raises(OSError) as caught:
        writer.put(_block(7))  # the caller hears of it, nothing is queued
    assert caught.value is disk.error
    writer.close()
    assert len(disk.blocks) == 2
    assert not writer.running
    # Reported once, the writer works again (here: the disk has room again).
    assert writer.take_error() is disk.error
    assert writer.take_error() is None
    disk.fail_at = None
    assert writer.put(_block(8))
    writer.close()
    assert _values(disk.blocks) == [0, 1, 8]


def test_abort_drops_what_waits_and_finishes_the_write_under_way():
    disk = _Disk()
    disk.gate.clear()
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    for value in range(4):
        writer.put(_block(value))
    assert disk.waiting.wait(5)  # block 0 is being written
    threading.Timer(0.05, disk.gate.set).start()
    writer.abort()
    assert not writer.running
    assert _values(disk.blocks) == [0]
    assert writer.pending_bytes == 0
    assert writer.error is None


def test_flush_can_give_up_waiting():
    disk = _Disk()
    disk.gate.clear()
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    writer.put(_block(1))
    assert not writer.flush(timeout=0.05)
    disk.gate.set()
    assert writer.flush(timeout=10)
    writer.close()


def test_writing_again_after_close():
    disk = _Disk()
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    writer.put(_block(1))
    writer.close()
    assert writer.put(_block(2))  # (a recording that goes on after a stop)
    writer.close()
    assert _values(disk.blocks) == [1, 2]
    assert writer.written == 200


def test_an_idle_thread_ends_by_itself(monkeypatch):
    monkeypatch.setattr(background_writer, "IDLE_EXIT_S", 0.01)
    disk = _Disk()
    writer = BackgroundWriter(disk, max_bytes=1 << 20)
    writer.put(_block(1))
    assert writer.flush()
    thread = writer._thread
    if thread is not None:
        thread.join(5)
    assert not writer.running  # (a writer dropped without close leaks nothing)
    assert writer.put(_block(2))  # and starts again when needed
    writer.close()
    assert _values(disk.blocks) == [1, 2]

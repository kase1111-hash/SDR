"""
Blocks of samples written on a thread of their own.

A receiver's display loop must never wait for the disk: at 20 MS/s an I/Q
recording writes 160 MB a second, and writing one display frame's block took
20-25 ms of the frame's 33 ms (over 100 ms while the disk caught up), so
the frames fell behind. :class:`BackgroundWriter` takes each block (a copy:
the caller may reuse its array at once) into a bounded queue and hands the
blocks, in order, to a write function on a thread of its own; file writes
release the GIL, so they run alongside the caller.

Nothing is lost silently:

* a block that doesn't fit in the queue (the disk can't keep up) is
  refused: :meth:`BackgroundWriter.put` returns False and the caller counts
  it as lost and says so (a block is always taken while nothing waits, so
  one larger than the queue isn't refused for its size alone);
* the first error the write function raises (a full disk) stops the
  writer: what is still queued is dropped, and the error waits in
  :attr:`BackgroundWriter.error` for the caller to report on its own thread
  (:meth:`~BackgroundWriter.put` raises it too);
* :meth:`~BackgroundWriter.flush` waits until everything queued is
  written, :meth:`~BackgroundWriter.close` also ends the thread (a later
  :meth:`~BackgroundWriter.put` starts it again), and
  :meth:`~BackgroundWriter.abort` drops what waits. A thread left idle ends
  by itself after :data:`IDLE_EXIT_S`, so a writer dropped without
  :meth:`~BackgroundWriter.close` leaves no thread behind.

No Qt. The main window's I/Q recording (its temporary take) and the second
receiver's recording write through it.

Thread safety: one thread (the caller's) puts, flushes, closes and aborts;
the writer's own thread only calls the write function. The write function
must not need the caller's thread (it runs on the writer's).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Deque, Optional

import numpy as np

#: Seconds an idle writer thread waits for a block before it ends (the next
#: :meth:`BackgroundWriter.put` starts another).
IDLE_EXIT_S = 2.0


class BackgroundWriter:
    """Writes blocks of samples in order on a thread of its own, from a
    queue that holds at most ``max_bytes`` (see the module docstring).

    Args:
        write: Called on the writer thread with each block (a C-contiguous
            array), in the order they were put; it writes all of it or
            raises. The first exception stops the writer.
        max_bytes: Most bytes queued (counting the block being written).
        name: The writer thread's name.
    """

    def __init__(
        self,
        write: Callable[[np.ndarray], Any],
        max_bytes: int,
        name: str = "sample writer",
    ) -> None:
        self._write = write
        self._max_bytes = max(0, int(max_bytes))
        self._name = name
        self._cond = threading.Condition(threading.Lock())
        self._queue: Deque[np.ndarray] = deque()
        self._pending = 0  # bytes queued, with the block being written
        self._busy = False  # a block is being written
        self._stop = False  # end the thread once the queue is empty
        self._error: Optional[BaseException] = None
        self._written = 0
        self._thread: Optional[threading.Thread] = None

    # -- state -----------------------------------------------------------

    @property
    def max_bytes(self) -> int:
        """Most bytes the queue holds (a lone larger block is taken too)."""
        return self._max_bytes

    @property
    def pending_bytes(self) -> int:
        """Bytes put but not written yet (with the block being written)."""
        with self._cond:
            return self._pending

    @property
    def written(self) -> int:
        """Items (samples) of the blocks written in full."""
        with self._cond:
            return self._written

    @property
    def error(self) -> Optional[BaseException]:
        """The error that stopped the writer (None while it works)."""
        with self._cond:
            return self._error

    @property
    def running(self) -> bool:
        """Whether the writer's thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def take_error(self) -> Optional[BaseException]:
        """The error that stopped the writer (None if none), forgotten: the
        caller reports it once, and a later :meth:`put` writes again."""
        with self._cond:
            error, self._error = self._error, None
            return error

    # -- the caller's side -------------------------------------------------

    def put(self, block: np.ndarray, copy: bool = True) -> bool:
        """Queue ``block`` to be written after the ones before it.

        The writer gets a copy (``copy=False``: the caller hands the array
        over and never touches it again). True when queued; False when the
        queue has no room for it (nothing is queued: the caller counts the
        block as lost). Raises the error that stopped the writer, if one
        did (nothing is queued)."""
        size = int(np.asarray(block).nbytes)
        with self._cond:
            if self._error is not None:
                raise self._error
            if size == 0:
                return True
            if self._pending and self._pending + size > self._max_bytes:
                return False
        # (Copied outside the lock: the queue only shrinks meanwhile, and
        # only this thread puts.)
        data = np.array(block, order="C") if copy else np.ascontiguousarray(block)
        with self._cond:
            if self._error is not None:  # (it failed meanwhile)
                raise self._error
            self._queue.append(data)
            self._pending += size
            if self._thread is None:
                self._stop = False
                self._thread = threading.Thread(
                    target=self._run, name=self._name, daemon=True
                )
                self._thread.start()
            else:
                self._cond.notify_all()
        return True

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until every block put is written, or the writer stopped on
        an error (see :attr:`error`). False when ``timeout`` seconds passed
        first."""
        with self._cond:
            return self._cond.wait_for(
                lambda: self._error is not None or (not self._queue and not self._busy),
                timeout,
            )

    def close(self, timeout: Optional[float] = None) -> bool:
        """Write everything put, then end the thread. False when ``timeout``
        passed first (the thread then goes on writing)."""
        if not self.flush(timeout):
            return False
        self._end()
        return True

    def abort(self) -> None:
        """Drop the blocks that wait, let the write under way finish, and
        end the thread."""
        with self._cond:
            while self._queue:
                self._pending -= self._queue.popleft().nbytes
        self._end()

    def _end(self) -> None:
        with self._cond:
            thread = self._thread
            self._stop = True
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    # -- the writer's thread -------------------------------------------------

    def _run(self) -> None:
        me = threading.current_thread()
        while True:
            with self._cond:
                if not self._queue and not self._stop:
                    self._cond.wait_for(lambda: self._queue or self._stop, IDLE_EXIT_S)
                if not self._queue:
                    # Stopped, or idle: the next put starts a thread again.
                    if self._thread is me:
                        self._thread = None
                    self._cond.notify_all()
                    return
                block = self._queue.popleft()
                self._busy = True
            try:
                self._write(block)
            except Exception as e:  # (any: the caller reports it)
                with self._cond:
                    self._error = e
                    self._busy = False
                    self._queue.clear()
                    self._pending = 0
                    if self._thread is me:
                        self._thread = None
                    self._cond.notify_all()
                return
            with self._cond:
                self._busy = False
                self._pending -= block.nbytes
                self._written += len(block)
                self._cond.notify_all()

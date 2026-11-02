"""
A display loop's per-frame work on a thread of its own, one frame behind.

The second receiver shares the GUI thread with the main window: its channel
filter, demodulator and plot spectrum took a quarter of each 33 ms frame.
:class:`FramePipeline` runs such a frame's job on a worker thread while the
GUI thread goes on (painting, the main window's frame): each display frame
collects the last frame's result, waiting only while that job still runs,
submits its own job and presents the result it collected. One job is in
flight at a time (the queue holds one), so the results come back in the
order the frames came, and the loop never runs more than a frame ahead of
the worker: a worker that falls behind slows the frames (which the display
loop's pacer then sees), it never piles up work or drops a frame's audio.

The job must not touch Qt (it runs on the worker); everything it changes
belongs to it while it runs, so the GUI thread collects (or waits for) the
job in flight before it changes those objects itself (a reset, new FFT
settings). Python's GIL lets the job run alongside the GUI thread only
while NumPy works without it (large FFTs and array operations), so the gain
is part of the job's time, not all of it.

No Qt. Thread safety: one thread (the display loop's) submits, collects and
closes.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

#: Seconds an idle worker thread waits for a job before it ends (the next
#: :meth:`FramePipeline.submit` starts another).
IDLE_EXIT_S = 2.0


class FramePipeline:
    """Runs one frame's job at a time on a worker thread (see the module
    docstring).

    Args:
        name: The worker thread's name.
    """

    def __init__(self, name: str = "frame worker") -> None:
        self._name = name
        self._cond = threading.Condition(threading.Lock())
        self._job: Optional[Callable[[], Any]] = None  # waiting to run
        self._pending = False  # submitted, its result not collected yet
        self._done = False  # its result is ready
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    @property
    def pending(self) -> bool:
        """Whether a job was submitted whose result isn't collected yet."""
        with self._cond:
            return self._pending

    @property
    def running(self) -> bool:
        """Whether the worker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def ready(self) -> bool:
        """Whether the job submitted last is done (its result waits to be
        collected)."""
        with self._cond:
            return self._pending and self._done

    def submit(self, job: Callable[[], Any]) -> None:
        """Run ``job`` (no arguments) on the worker. Collect the last job's
        result first: one job at a time (RuntimeError otherwise)."""
        with self._cond:
            if self._pending:
                raise RuntimeError("collect the last frame's result first")
            self._job = job
            self._pending = True
            self._done = False
            if self._thread is None:
                self._stop = False
                self._thread = threading.Thread(
                    target=self._run, name=self._name, daemon=True
                )
                self._thread.start()
            else:
                self._cond.notify_all()

    def collect(self) -> Any:
        """The result of the job submitted last, waiting for it while it
        runs (None when nothing was submitted). Raises what the job raised."""
        with self._cond:
            if not self._pending:
                return None
            self._cond.wait_for(lambda: self._done)
            self._pending = False
            result, error = self._result, self._error
            self._result = self._error = None
        if error is not None:
            raise error
        return result

    def close(self) -> None:
        """Let the job in flight finish (its result stays to be collected)
        and end the worker thread."""
        with self._cond:
            self._cond.wait_for(lambda: not self._pending or self._done)
            thread = self._thread
            self._stop = True
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _run(self) -> None:
        me = threading.current_thread()
        while True:
            with self._cond:
                if self._job is None and not self._stop:
                    self._cond.wait_for(
                        lambda: self._job is not None or self._stop, IDLE_EXIT_S
                    )
                if self._job is None:
                    # Closed, or idle: the next submit starts a thread again.
                    if self._thread is me:
                        self._thread = None
                    self._cond.notify_all()
                    return
                job, self._job = self._job, None
            try:
                result, error = job(), None
            except Exception as e:  # (any: the display loop hears of it)
                result, error = None, e
            with self._cond:
                self._result, self._error, self._done = result, error, True
                self._cond.notify_all()

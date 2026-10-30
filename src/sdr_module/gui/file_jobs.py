"""
Long file work off the GUI thread, with progress and Cancel.

:func:`run_job` runs a work function in a worker thread while a progress
dialog shows how far it got (only if it takes more than a moment) and
offers Cancel; the window stays responsive meanwhile (the spectrum keeps
running). It returns when the work is done, so callers read like plain
function calls.

The work functions here read recordings a chunk at a time, never the whole
file at once:

* :func:`load_recording` streams a file into the recording buffer
  (File > Convert Recording);
* :func:`write_recording` writes the buffer in another format
  (File > Save Recording);
* :func:`decode_recording` runs a live decoder over a whole file, much
  faster than real time (File > Decode Recording).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

from ..dsp.recording import FileFormat, IQRecorder, SampleFormat
from .iq_files import IqFileOptions

try:
    from PyQt6.QtCore import QEvent, QEventLoop, QObject, Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import QApplication, QProgressDialog

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

logger = logging.getLogger(__name__)

#: Samples read (and converted) per step.
CHUNK_SAMPLES = 1 << 20

#: ``report(fraction)`` (0..1): raises :class:`Cancelled` once Cancel was
#: pressed, so the work stops at its next report.
Report = Callable[[float], None]


class Cancelled(Exception):
    """Cancel was pressed."""


@dataclass
class JobResult:
    ok: bool = False
    cancelled: bool = False
    error: Optional[BaseException] = None
    value: Any = None


@dataclass
class LoadResult:
    samples: int = 0
    #: Samples that weren't numbers (NaN or infinity): a wrong format.
    nonfinite: int = 0
    max_abs: float = 0.0


@dataclass
class DecodeResult:
    #: (message, seconds into the file) of every message, valid or not.
    messages: List[Tuple[Any, float]] = field(default_factory=list)
    seconds: float = 0.0  # of the file decoded

    @property
    def valid(self) -> int:
        return sum(1 for msg, _t in self.messages if getattr(msg, "valid", True))

    @property
    def failed(self) -> int:
        return len(self.messages) - self.valid


def _chunks(options: IqFileOptions, report: Report) -> Iterable[np.ndarray]:
    """The file's samples, a chunk at a time, reporting progress."""
    with options.open_player() as player:
        total = max(1, player.total_samples)
        done = 0
        report(0.0)
        while True:
            chunk = player.read(CHUNK_SAMPLES)
            if not len(chunk):
                break
            done += len(chunk)
            yield chunk
            report(min(1.0, done / total))


def load_recording(
    options: IqFileOptions, append: Callable[[np.ndarray], None], report: Report
) -> LoadResult:
    """Read the whole file (as ``options`` say) into ``append``, a chunk at
    a time, counting samples that aren't numbers."""
    result = LoadResult()
    for chunk in _chunks(options, report):
        finite = np.isfinite(chunk)
        bad = int(chunk.size - np.count_nonzero(finite))
        if bad:
            result.nonfinite += bad
        with np.errstate(all="ignore"):
            good = np.abs(chunk[finite]) if bad else np.abs(chunk)
            if good.size:
                result.max_abs = max(result.max_abs, float(good.max()))
        append(chunk)
        result.samples += len(chunk)
    return result


def write_recording(
    chunks: Iterable[np.ndarray],
    total: int,
    filename: str,
    file_format: FileFormat,
    sample_format: SampleFormat,
    sample_rate: float,
    center_frequency: float,
    report: Report,
    start_time: str = "",
) -> int:
    """Write ``chunks`` (``total`` samples) to ``filename``; a cancelled or
    failed write leaves no partial file behind. ``start_time`` (UTC, ISO
    8601) is when the samples were recorded, for SigMF's ``core:datetime``
    (else the time of writing). Returns the samples written."""
    if file_format is FileFormat.WAV and not sample_rate > 0:
        # (Its header must hold one: nothing is written.)
        raise ValueError(
            "a WAV file needs the sample rate, and this recording doesn't say "
            "it. Save again and enter it, or choose another format (SigMF can "
            "leave it unknown)."
        )
    recorder = IQRecorder(
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        sample_format=sample_format,
        file_format=file_format,
    )
    written = 0
    opened = complete = False
    try:
        recorder.start(filename)
        opened = True
        if start_time:
            recorder.set_metadata(start_time=start_time)
        try:
            report(0.0)
            for chunk in chunks:
                recorder.write(chunk)
                written += len(chunk)
                report(min(1.0, written / max(1, total)))
        except BaseException:
            _stop_quietly(recorder)  # (the file is removed below)
            raise
        # Flushes and closes the file and writes SigMF's metadata: a full
        # disk can fail here too.
        recorder.stop()
        complete = True
    finally:
        if opened and not complete:
            _remove_partial(filename, file_format)
    return written


def _stop_quietly(recorder: IQRecorder) -> None:
    """Close a write that failed (its own error is the one reported)."""
    try:
        recorder.stop()
    except Exception as e:
        logger.debug(f"Closing the failed recording write also failed: {e}")


def _remove_partial(filename: str, file_format: FileFormat) -> None:
    names = [filename]
    if file_format is FileFormat.SIGMF:
        stem = os.path.splitext(filename)[0]
        names = [stem + ".sigmf-data", stem + ".sigmf-meta"]
    for name in names:
        try:
            os.remove(name)
        except OSError:
            pass


def decode_recording(
    options: IqFileOptions,
    protocol: str,
    offset_hz: float,
    report: Report,
    found: Optional[Callable[[List[Tuple[Any, float]]], None]] = None,
) -> DecodeResult:
    """Run the live decoder for ``protocol`` over the whole file, its
    channel ``offset_hz`` from the file's center, a chunk at a time (as
    fast as the computer goes). ``found`` gets each chunk's messages with
    their time in the file as they come."""
    from ..dsp.live_decoders import create_live_decoder

    decoder = create_live_decoder(protocol, options.sample_rate, offset_hz)
    status = decoder.status
    if status.problem:
        raise ValueError(status.problem)
    result = DecodeResult()
    rate = float(options.sample_rate)
    position = 0
    for chunk in _chunks(options, report):
        position += len(chunk)
        messages = decoder.process(chunk)
        if messages:
            stamped = [(msg, position / rate) for msg in messages]
            result.messages.extend(stamped)
            if found is not None:
                found(stamped)
    # What the decoder still holds at the end of the file (e.g. the last
    # line of Morse code), as the command line's decode does.
    tail = decoder.flush()
    if tail:
        stamped = [(msg, position / rate) for msg in tail]
        result.messages.extend(stamped)
        if found is not None:
            found(stamped)
    result.seconds = position / rate
    return result


if HAS_PYQT6:

    class _Worker(QThread):
        progress = pyqtSignal(float)
        partial = pyqtSignal(object)

        def __init__(self, work: Callable[..., Any], wants_partial: bool):
            super().__init__()
            self._work = work
            self._wants_partial = wants_partial
            self._cancel = False
            self.result = JobResult()

        def cancel(self) -> None:
            self._cancel = True

        def _report(self, fraction: float) -> None:
            if self._cancel:
                raise Cancelled()
            self.progress.emit(float(fraction))

        def run(self) -> None:
            try:
                if self._wants_partial:
                    value = self._work(self._report, self.partial.emit)
                else:
                    value = self._work(self._report)
            except Cancelled:
                self.result = JobResult(cancelled=True)
            except BaseException as e:  # reported to the caller
                logger.debug("File job failed", exc_info=True)
                self.result = JobResult(error=e)
            else:
                self.result = JobResult(ok=not self._cancel, value=value)
                self.result.cancelled = self._cancel

    class _HoldInput(QObject):
        """Keyboard and mouse input waits while a job runs: the window's
        event loop goes on (the spectrum keeps running), but a click on
        Record or a shortcut must not start something under the job's feet
        (a recording going into the buffer being converted). Once the
        progress dialog is up its modality does this; before (the first
        moment) this filter swallows the input."""

        _INPUT = frozenset(
            {
                QEvent.Type.KeyPress,
                QEvent.Type.KeyRelease,
                QEvent.Type.ShortcutOverride,
                QEvent.Type.Shortcut,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonRelease,
                QEvent.Type.MouseButtonDblClick,
                QEvent.Type.Wheel,
                QEvent.Type.ContextMenu,
            }
        )

        def __init__(self, dialog: "QProgressDialog"):
            super().__init__(dialog)
            self._dialog = dialog

        def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt
            if event.type() not in self._INPUT or self._dialog.isVisible():
                return False
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()  # (taken as a key press: no shortcut fires)
            return True

    def run_job(
        parent: Any,
        title: str,
        label: str,
        work: Callable[..., Any],
        on_partial: Optional[Callable[[Any], None]] = None,
        show_after_ms: int = 400,
    ) -> JobResult:
        """Run ``work(report)`` (or ``work(report, partial)`` with
        ``on_partial``, which then gets what ``partial(...)`` is called
        with, on the GUI thread) in a worker thread, with a progress dialog
        (shown once it takes longer than ``show_after_ms``) and Cancel.
        Keyboard and mouse input to the app waits meanwhile (see
        :class:`_HoldInput`). Returns once it is done."""
        worker = _Worker(work, on_partial is not None)
        dialog = QProgressDialog(label, "Cancel", 0, 1000, parent)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(max(0, int(show_after_ms)))
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        worker.progress.connect(lambda f: dialog.setValue(int(round(f * 1000))))
        if on_partial is not None:
            worker.partial.connect(on_partial)
        dialog.canceled.connect(worker.cancel)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        hold = _HoldInput(dialog)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(hold)
        try:
            worker.start()
            loop.exec()
            worker.wait()
            # Deliver what the worker queued just before it finished.
            loop.processEvents()
        finally:
            if app is not None:
                app.removeEventFilter(hold)
        dialog.close()
        dialog.deleteLater()
        result = worker.result
        worker.deleteLater()
        return result

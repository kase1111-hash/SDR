"""Every decoded message since Clear, kept on disk for Export.

The Decoder tab's table and Log show the newest 1,000 messages (more would
slow the window down), but Export saves the whole session: each message is
also appended, as the CSV row Export writes, to a spool file in the
system's temporary folder. Clear and quitting delete it; a session that
crashed leaves one behind, which the next session deletes when it starts
its own (a spool in use is locked, POSIX, or can't be deleted, Windows).

Qt-free: the panel (``decoder_panel.py``) owns a :class:`MessageSpool` and
says what it can't do (a full disk).
"""

import csv
import io
import logging
import os
import tempfile
import uuid
import weakref
from typing import Any, BinaryIO, Callable, Optional, Sequence

try:  # POSIX: a spool in use is locked
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_PREFIX = "sdr-decoded-"
_SUFFIX = ".csv"
#: Bytes copied per step when a spool is exported.
COPY_CHUNK = 1 << 20


def spool_folder() -> str:
    """The folder spools are written to (created if needed): a folder of
    the app's own in the system's temporary folder."""
    folder = os.path.join(tempfile.gettempdir(), "sdr-module-decoded")
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError:
        return tempfile.gettempdir()
    return folder


def _in_use(path: str) -> bool:
    """Whether a running session holds this spool (it keeps it locked,
    POSIX; on Windows an open file can't be deleted, which tells)."""
    if fcntl is None:
        return False
    try:
        with open(path, "rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        return True
    return False


def remove_stale_spools(folder: str) -> int:
    """Delete the spools of sessions that ended without cleaning up (a
    crash, a killed process); returns how many. Only this app's own files
    (``sdr-decoded-*.csv``) that no running session holds are touched."""
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    removed = 0
    for name in names:
        if not (name.startswith(_PREFIX) and name.endswith(_SUFFIX)):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or _in_use(path):
            continue
        try:
            os.remove(path)
        except OSError:  # in use (Windows) or not ours to delete
            continue
        removed += 1
        logger.info(f"Removed a stale decoded-message spool: {path}")
    return removed


def _close_and_remove(handle: Any, path: str) -> None:
    """Close ``handle`` (unlocking it) and delete ``path``, quietly: the
    spool's finalizer, run by :meth:`MessageSpool.discard` or at exit."""
    try:
        handle.close()
    except OSError:
        pass
    try:
        os.remove(path)
    except OSError:
        pass


class MessageSpool:
    """An append-only file of CSV rows: every message since :meth:`clear`.

    :meth:`append` writes each row through to the file (so a row is either
    there whole or not at all); once a write fails, e.g. on a full disk,
    the spool keeps what it has and takes no more until it is cleared
    (:attr:`problem` says why). :meth:`snapshot` and :meth:`copy_rows`
    copy what it held at one moment, from any thread, while rows go on
    being appended.

    Not thread-safe apart from :meth:`copy_rows`: append and clear from one
    thread (the GUI's).
    """

    def __init__(self, folder: Optional[str] = None):
        self._folder = folder
        self._handle: Optional[BinaryIO] = None
        self._path: Optional[str] = None
        self._finalizer: Optional[weakref.finalize] = None
        self._count = 0
        self._bytes = 0
        self._problem = ""
        self._buffer = io.StringIO()
        self._writer = csv.writer(self._buffer)

    @property
    def count(self) -> int:
        """Rows held (every row appended since :meth:`clear`, unless a
        write failed: then the ones before it)."""
        return self._count

    @property
    def nbytes(self) -> int:
        return self._bytes

    @property
    def path(self) -> Optional[str]:
        """The spool file (None until the first row)."""
        return self._path

    @property
    def problem(self) -> str:
        """Why rows are no longer kept ("" while they are)."""
        return self._problem

    def append(self, row: Sequence[Any]) -> bool:
        """Add a row (the values of one CSV line); False when it isn't
        kept (an earlier write failed, or this one did)."""
        if self._problem:
            return False
        self._buffer.seek(0)
        self._buffer.truncate()
        self._writer.writerow(row)
        data = self._buffer.getvalue().encode("utf-8")
        try:
            handle = self._open()
            handle.write(data)
            handle.flush()
        except OSError as exc:
            self._fail(exc)
            return False
        self._count += 1
        self._bytes += len(data)
        return True

    def _open(self) -> BinaryIO:
        if self._handle is not None:
            return self._handle
        folder = self._folder or spool_folder()
        os.makedirs(folder, exist_ok=True)
        remove_stale_spools(folder)
        path = os.path.join(folder, f"{_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:8]}")
        path += _SUFFIX
        handle = open(path, "xb")
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                pass  # (no locks on this file system: it still works)
        self._handle, self._path = handle, path
        # Deleted however the session ends without discard(): at exit too.
        self._finalizer = weakref.finalize(self, _close_and_remove, handle, path)
        return handle

    def _fail(self, exc: OSError) -> None:
        """A write failed: keep the rows written in full (cut a partial one
        off) and take no more."""
        reason = exc.strerror or str(exc) or type(exc).__name__
        where = os.path.dirname(self._path) if self._path else self._folder
        self._problem = f"{reason} in {where or spool_folder()}"
        if self._handle is not None:
            try:
                self._handle.truncate(self._bytes)
            except (OSError, ValueError):
                pass
        logger.warning(
            "Decoder: could not keep every message for Export (writing the "
            f"message spool failed: {self._problem})"
        )

    def snapshot(self) -> int:
        """Bytes the spool holds now (what :meth:`copy_rows` copies)."""
        return self._bytes

    def copy_rows(
        self,
        target: BinaryIO,
        length: int,
        progress: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Copy the first ``length`` bytes of the spool (a :meth:`snapshot`
        taken before) to ``target``, through a handle of its own: rows
        appended meanwhile don't get in the way. Raises OSError."""
        if not length or not self._path:
            return
        done = 0
        with open(self._path, "rb") as source:
            while done < length:
                chunk = source.read(min(COPY_CHUNK, length - done))
                if not chunk:
                    raise OSError(f"the message spool ended early ({self._path})")
                target.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done / length)

    def clear(self) -> None:
        """Forget every row and delete the file (a full disk is forgotten
        too: the next row tries again)."""
        self.discard()
        self._problem = ""

    def discard(self) -> None:
        """Delete the file (the app is closing, or Clear)."""
        finalizer, self._finalizer = self._finalizer, None
        if finalizer is not None:
            finalizer()
        self._handle = None
        self._path = None
        self._count = 0
        self._bytes = 0

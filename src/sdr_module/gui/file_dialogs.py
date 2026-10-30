"""
File dialogs that remember where you were.

Each kind of file has its own folder memory: a dialog for I/Q recordings
opens where the last I/Q recording was opened or saved, a screenshot dialog
where the last screenshot went, and so on. The folders are kept in the
settings under ``recording/last_folder/<kind>``. Until a kind has been used
it starts in its default folder: ``Documents/SDR Recordings`` (created the
first time something is saved there), ``Pictures`` for screenshots.

The folder automatic recordings go to (audio recordings, auto-saved I/Q
takes) is :func:`recordings_folder`, ``recording/folder`` in the settings.

Usage::

    from . import file_dialogs

    path, _filter = file_dialogs.open_file(
        self, self._settings, "iq", "Open Recording", "SigMF (*.sigmf-data)"
    )
    path, chosen = file_dialogs.save_file(
        self, self._settings, "screenshot", "Save Screenshot",
        "sdr_100.100MHz_20261030_120000.png", "PNG (*.png)",
    )

``settings`` is anything with ``get_str(key, default)`` and ``set(key,
value)`` (:class:`~.settings_store.GuiSettings`). The dialogs are Qt's
static ``QFileDialog`` functions, so tests can patch those as before.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

try:
    from PyQt6.QtCore import QStandardPaths, QUrl
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import QFileDialog

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

logger = logging.getLogger(__name__)

#: Name of the app's own folder in Documents.
RECORDINGS_SUBFOLDER = "SDR Recordings"
#: Settings key prefix of the remembered folders (one per kind).
LAST_FOLDER_KEY = "recording/last_folder/"
#: Settings key of the folder automatic recordings go to.
RECORDINGS_FOLDER_KEY = "recording/folder"

#: Stand-ins for the user's folders ("documents", "pictures"), e.g. a
#: temporary folder while the tests run.
LOCATION_OVERRIDES: Dict[str, str] = {}

#: Where each kind of file starts until a folder has been used for it:
#: "recordings" (Documents/SDR Recordings), "pictures" or "documents".
KIND_DEFAULTS: Dict[str, str] = {
    "iq": "recordings",
    "audio": "recordings",
    "decode": "recordings",
    "screenshot": "pictures",
    "channels": "documents",
}


def _standard_location(location: Any) -> str:
    if HAS_PYQT6:
        try:
            path = QStandardPaths.writableLocation(location)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"No standard location {location}: {e}")
            path = ""
        if path:
            return path
    return os.path.expanduser("~")


def documents_folder() -> str:
    """The user's Documents folder (their home folder if there is none)."""
    if LOCATION_OVERRIDES.get("documents"):
        return LOCATION_OVERRIDES["documents"]
    if not HAS_PYQT6:  # pragma: no cover
        return os.path.expanduser("~")
    return _standard_location(QStandardPaths.StandardLocation.DocumentsLocation)


def pictures_folder() -> str:
    """The user's Pictures folder (their home folder if there is none)."""
    if LOCATION_OVERRIDES.get("pictures"):
        return LOCATION_OVERRIDES["pictures"]
    if not HAS_PYQT6:  # pragma: no cover
        return os.path.expanduser("~")
    return _standard_location(QStandardPaths.StandardLocation.PicturesLocation)


def default_recordings_folder() -> str:
    """``Documents/SDR Recordings``."""
    return os.path.join(documents_folder(), RECORDINGS_SUBFOLDER)


def default_folder(kind: str) -> str:
    """Where ``kind`` starts before a folder has been used for it."""
    where = KIND_DEFAULTS.get(kind, "recordings")
    if where == "pictures":
        return pictures_folder()
    if where == "documents":
        return documents_folder()
    return default_recordings_folder()


def ensure_folder(folder: str) -> bool:
    """Create ``folder`` (and its parents) if needed; False if it can't be."""
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        logger.debug(f"Can't create {folder}: {e}")
        return False
    return os.path.isdir(folder)


def remembered_folder(settings: Any, kind: str) -> str:
    """The folder last used for ``kind``, if it still exists ("" if not)."""
    try:
        folder = str(settings.get_str(LAST_FOLDER_KEY + kind, "") or "")
    except Exception:  # pragma: no cover - defensive
        return ""
    return folder if folder and os.path.isdir(folder) else ""


def remember_folder(settings: Any, kind: str, path: str) -> None:
    """Remember the folder of ``path`` (a file or a folder) for ``kind``."""
    if not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    try:
        settings.set(LAST_FOLDER_KEY + kind, folder)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Could not remember the {kind} folder: {e}")


def start_folder(settings: Any, kind: str, create: bool = False) -> str:
    """The folder a ``kind`` dialog opens in: the one last used, else the
    default (created first with ``create``, when a file will be saved
    there). A default that doesn't exist falls back to its parent."""
    folder = remembered_folder(settings, kind)
    if folder:
        return folder
    folder = default_folder(kind)
    if os.path.isdir(folder) or (create and ensure_folder(folder)):
        return folder
    parent = os.path.dirname(folder)
    return parent if os.path.isdir(parent) else os.path.expanduser("~")


def recordings_folder(settings: Any) -> str:
    """Where automatic recordings go (``recording/folder``; default
    ``Documents/SDR Recordings``). Not created here: see :func:`ensure_folder`."""
    try:
        folder = str(settings.get_str(RECORDINGS_FOLDER_KEY, "") or "")
    except Exception:  # pragma: no cover - defensive
        folder = ""
    return folder or default_recordings_folder()


def set_recordings_folder(settings: Any, folder: str) -> None:
    settings.set(RECORDINGS_FOLDER_KEY, str(folder))


def unique_path(folder: str, name: str) -> str:
    """``folder/name``, or ``folder/<stem>_2<ext>`` (``_3``...) when taken.
    A SigMF name counts as taken when either of its two files exists."""
    stem, ext = _split_name(name)
    candidate = os.path.join(folder, name)
    number = 2
    while _taken(candidate):
        candidate = os.path.join(folder, f"{stem}_{number}{ext}")
        number += 1
    return candidate


def _split_name(name: str) -> Tuple[str, str]:
    lower = name.lower()
    for ext in (".sigmf-data", ".sigmf-meta"):
        if lower.endswith(ext):
            return name[: -len(ext)], name[-len(ext) :]
    return os.path.splitext(name)


def _taken(path: str) -> bool:
    stem, ext = _split_name(path)
    if ext.lower() in (".sigmf-data", ".sigmf-meta"):
        return os.path.exists(stem + ".sigmf-data") or os.path.exists(
            stem + ".sigmf-meta"
        )
    return os.path.exists(path)


def open_file(
    parent: Any,
    settings: Any,
    kind: str,
    caption: str,
    filters: str,
    selected_filter: str = "",
) -> Tuple[str, str]:
    """``QFileDialog.getOpenFileName`` in ``kind``'s folder; remembers the
    folder of the file chosen. Returns ``(path, filter)``, path "" if
    cancelled."""
    path, chosen = QFileDialog.getOpenFileName(
        parent, caption, start_folder(settings, kind), filters, selected_filter
    )
    if path:
        remember_folder(settings, kind, path)
    return path, chosen


def save_file(
    parent: Any,
    settings: Any,
    kind: str,
    caption: str,
    file_name: str,
    filters: str,
    selected_filter: str = "",
) -> Tuple[str, str]:
    """``QFileDialog.getSaveFileName`` suggesting ``file_name`` in ``kind``'s
    folder (a default folder is created first); remembers the folder of
    the file chosen. Returns ``(path, filter)``, path "" if cancelled."""
    folder = start_folder(settings, kind, create=True)
    path, chosen = QFileDialog.getSaveFileName(
        parent, caption, os.path.join(folder, file_name), filters, selected_filter
    )
    if path:
        remember_folder(settings, kind, path)
    return path, chosen


def choose_folder(
    parent: Any,
    settings: Any,
    kind: str,
    caption: str,
    current: str = "",
    remember: bool = True,
) -> str:
    """``QFileDialog.getExistingDirectory`` starting at ``current`` (or
    ``kind``'s folder); remembers the choice for ``kind`` (unless
    ``remember`` is False, e.g. in a dialog that can still be cancelled).
    "" if cancelled."""
    start = current if current and os.path.isdir(current) else ""
    folder = QFileDialog.getExistingDirectory(
        parent, caption, start or start_folder(settings, kind, create=True)
    )
    if folder and remember:
        remember_folder(settings, kind, folder)
    return folder


def show_in_folder(path: str) -> bool:
    """Open the folder holding ``path`` (or the folder ``path``) in the
    system's file manager. False if that didn't work."""
    if not HAS_PYQT6 or not path:
        return False
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    try:
        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(folder)))
    except Exception as e:  # pragma: no cover - platform-dependent
        logger.warning(f"Could not open {folder}: {e}")
        return False


def display_path(path: str) -> str:
    """``path`` with the home folder shown as ``~`` (for messages)."""
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


__all__ = [
    "KIND_DEFAULTS",
    "RECORDINGS_SUBFOLDER",
    "choose_folder",
    "default_folder",
    "default_recordings_folder",
    "display_path",
    "documents_folder",
    "ensure_folder",
    "open_file",
    "pictures_folder",
    "recordings_folder",
    "remember_folder",
    "remembered_folder",
    "save_file",
    "set_recordings_folder",
    "show_in_folder",
    "start_folder",
    "unique_path",
]

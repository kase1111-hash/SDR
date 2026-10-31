"""
Lightweight persistence for GUI-specific state.

Wraps QSettings so the rest of the GUI can read/write typed values
without caring about platform-specific storage locations.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List

try:
    from PyQt6.QtCore import QSettings

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    QSettings = None  # type: ignore

logger = logging.getLogger(__name__)

ORG = "SDR Module Team"
APP = "SDR Module"

# Frequency the GUI starts on when nothing is saved: a standard FM broadcast
# channel, and where demo mode's strongest station sits.
DEFAULT_FREQUENCY_HZ = 100.1e6


class GuiSettings:
    """Typed wrapper over QSettings for GUI state persistence."""

    def __init__(self):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        self._s = QSettings(ORG, APP)

    # ---- Simple typed getters ----
    def get_float(self, key: str, default: float) -> float:
        try:
            return float(self._s.value(key, default))
        except (TypeError, ValueError):
            return default

    def get_int(self, key: str, default: int) -> int:
        try:
            return int(self._s.value(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool) -> bool:
        raw = self._s.value(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() in ("1", "true", "yes", "on")
        return bool(raw)

    def get_str(self, key: str, default: str) -> str:
        v = self._s.value(key, default)
        return str(v) if v is not None else default

    def set(self, key: str, value: Any) -> None:
        self._s.setValue(key, value)

    def sync(self) -> None:
        self._s.sync()

    # ---- Bookmark helpers (stored as JSON string) ----
    def get_bookmarks(self) -> List[Dict[str, Any]]:
        raw = self._s.value("bookmarks", "[]")
        try:
            data = json.loads(raw if isinstance(raw, str) else "[]")
        except (ValueError, TypeError):
            logger.warning("Corrupt bookmarks in settings; resetting")
            return []
        # (One damaged entry must not stop the app starting.)
        return usable_bookmarks(data)

    def set_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> None:
        self._s.setValue("bookmarks", json.dumps(bookmarks))

    # ---- First-run flag ----
    def is_first_run(self) -> bool:
        return self.get_bool("first_run_done", False) is False

    def mark_first_run_done(self) -> None:
        self.set("first_run_done", True)

    # ---- Window geometry ----
    def save_geometry(self, name: str, geometry: bytes) -> None:
        self._s.setValue(f"geom/{name}", geometry)

    def load_geometry(self, name: str):
        return self._s.value(f"geom/{name}")


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _examples(entries: List[Any]) -> str:
    """``"null, 1, ..."``: the first entries, short, for the log."""
    shown = []
    for entry in entries[:3]:
        try:
            text = json.dumps(entry)
        except (TypeError, ValueError):
            text = repr(entry)
        shown.append(text if len(text) <= 60 else text[:57] + "...")
    return ", ".join(shown) + (", ..." if len(entries) > 3 else "")


def usable_bookmarks(data: Any) -> List[Dict[str, Any]]:
    """The bookmarks of a saved list, safe to list, tune and scan.

    A damaged settings file can hold anything, and one bad entry stopped the
    app starting. An entry that isn't a bookmark (null, a number, a text) is
    skipped; a bookmark whose frequency isn't a number (missing, "abc", NaN,
    too big for a float) stays in the list at 0 Hz, to be edited or
    removed. Each kind of damage is logged once (Tools > Error History).
    """
    if data is None:
        return []
    if not isinstance(data, list):
        logger.warning(
            "Ignored the saved bookmarks: not a list (%s).", _examples([data])
        )
        return []
    kept: List[Dict[str, Any]] = []
    dropped: List[Any] = []
    repaired: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            dropped.append(entry)
            continue
        if not _finite_number(entry.get("freq_hz")):
            repaired.append(dict(entry))
            entry["freq_hz"] = 0.0
        kept.append(entry)
    if dropped:
        noun = "entry" if len(dropped) == 1 else "entries"
        logger.warning(
            "Ignored %d damaged %s of the saved bookmarks (%s): not a bookmark.",
            len(dropped),
            noun,
            _examples(dropped),
        )
    if repaired:
        noun = "bookmark has" if len(repaired) == 1 else "bookmarks have"
        logger.warning(
            "%d saved %s no usable frequency (%s): listed at 0 Hz, to be "
            "edited or removed.",
            len(repaired),
            noun,
            _examples(repaired),
        )
    return kept

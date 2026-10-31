"""
Lightweight persistence for GUI-specific state.

Wraps QSettings so the rest of the GUI can read/write typed values
without caring about platform-specific storage locations.
"""

from __future__ import annotations

import json
import logging
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

#: Qt's integers (a spin box, a slider) are 32-bit: a saved number outside
#: this range is no value the app wrote.
INT_RANGE = (-(2**31), 2**31 - 1)
#: How a saved yes/no reads (QSettings' INI files hold text).
_TRUE_TEXT = ("1", "true", "yes", "on")
_FALSE_TEXT = ("0", "false", "no", "off")


class GuiSettings:
    """Typed wrapper over QSettings for GUI state persistence."""

    def __init__(self):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        self._s = QSettings(ORG, APP)
        self._damaged_keys: set[str] = set()  # (logged once each)

    def _damaged(self, key: str, raw: Any, default: Any) -> Any:
        """A saved value that can't be one the app wrote (a damaged or
        hand-edited settings file): the default instead, said once per key
        in the log (Tools > Error History)."""
        if key not in self._damaged_keys:
            self._damaged_keys.add(key)
            logger.warning(
                "Ignoring the damaged setting %s=%r: using %r instead.",
                key,
                raw,
                default,
            )
        return default

    # ---- Simple typed getters ----
    # A value that isn't of the key's type (or is out of reach: not a
    # finite number, a 64-bit integer) gives the default, logged once.
    def get_float(self, key: str, default: float) -> float:
        raw = self._s.value(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return self._damaged(key, raw, default)
        if not -float("inf") < value < float("inf") and self._s.contains(key):
            return self._damaged(key, raw, default)
        return value

    def get_int(self, key: str, default: int) -> int:
        raw = self._s.value(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            return self._damaged(key, raw, default)
        if not INT_RANGE[0] <= value <= INT_RANGE[1]:
            return self._damaged(key, raw, default)
        return value

    def get_bool(self, key: str, default: bool) -> bool:
        raw = self._s.value(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int) and raw in (0, 1):
            return bool(raw)
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in _TRUE_TEXT:
                return True
            if text in _FALSE_TEXT:
                return False
        return self._damaged(key, raw, default)

    def get_str(self, key: str, default: str) -> str:
        v = self._s.value(key, default)
        if v is None:
            return default
        if isinstance(v, (list, tuple, dict, bytes, bytearray)):
            return self._damaged(key, v, default)  # (e.g. "a, b": a list)
        return str(v)

    def set(self, key: str, value: Any) -> None:
        self._s.setValue(key, value)

    def sync(self) -> None:
        self._s.sync()

    # ---- Bookmark helpers (stored as JSON string) ----
    def get_bookmarks(self) -> List[Dict[str, Any]]:
        raw = self._s.value("bookmarks", "[]")
        try:
            data = json.loads(raw if isinstance(raw, str) else "[]")
            return data if isinstance(data, list) else []
        except (ValueError, TypeError):
            logger.warning("Corrupt bookmarks in settings; resetting")
            return []

    def set_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> None:
        self._s.setValue("bookmarks", json.dumps(bookmarks))

    # ---- First-run flag ----
    def is_first_run(self) -> bool:
        # (A damaged flag that is there at all: the app has run before.)
        done = self._s.contains("first_run_done")
        return self.get_bool("first_run_done", done) is False

    def mark_first_run_done(self) -> None:
        self.set("first_run_done", True)

    # ---- Window geometry ----
    def save_geometry(self, name: str, geometry: bytes) -> None:
        self._s.setValue(f"geom/{name}", geometry)

    def load_geometry(self, name: str):
        return self._s.value(f"geom/{name}")

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
            return data if isinstance(data, list) else []
        except (ValueError, TypeError):
            logger.warning("Corrupt bookmarks in settings; resetting")
            return []

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

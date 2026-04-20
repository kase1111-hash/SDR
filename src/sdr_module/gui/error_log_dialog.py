"""
Error history viewer.

Captures warning/error-level log records and surfaces them in a dialog so
users can review transient status-bar messages they may have missed.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Deque, Tuple

try:
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class _HistoryHandler(logging.Handler):
    """Ring-buffer log handler that keeps the last N records."""

    def __init__(self, capacity: int = 500):
        super().__init__(level=logging.WARNING)
        self._buffer: Deque[Tuple[str, int, str]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        self._buffer.append((ts, record.levelno, self.format(record)))

    def snapshot(self):
        return list(self._buffer)

    def clear_history(self) -> None:
        self._buffer.clear()


_HISTORY = _HistoryHandler()
_HISTORY.setFormatter(logging.Formatter("%(name)s: %(message)s"))

# Install once on the root logger
_ROOT_INSTALLED = False


def install_history_handler() -> _HistoryHandler:
    """Install the error-history handler on the root logger (idempotent)."""
    global _ROOT_INSTALLED
    if not _ROOT_INSTALLED:
        logging.getLogger().addHandler(_HISTORY)
        _ROOT_INSTALLED = True
    return _HISTORY


class ErrorLogDialog(QDialog if HAS_PYQT6 else object):
    """Viewer for recent warning/error messages."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self.setWindowTitle("Error History")
        self.resize(700, 420)

        layout = QVBoxLayout(self)
        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        layout.addWidget(self._view)

        self._refresh()

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        btns.addButton(refresh_btn, QDialogButtonBox.ButtonRole.ActionRole)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        btns.addButton(clear_btn, QDialogButtonBox.ButtonRole.DestructiveRole)
        layout.addWidget(btns)

    def _refresh(self):
        records = _HISTORY.snapshot()
        if not records:
            self._view.setPlainText("(no warnings or errors logged)")
            return
        lines = []
        for ts, lvl, msg in records:
            tag = "ERROR" if lvl >= logging.ERROR else "WARN"
            lines.append(f"[{ts}] {tag}  {msg}")
        self._view.setPlainText("\n".join(lines))

    def _clear(self):
        _HISTORY.clear_history()
        self._refresh()

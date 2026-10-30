"""
The status bar's recent messages.

Status-bar messages clear themselves after a few seconds. The window keeps
the last 20 in a :class:`StatusHistory`; a click on the message area opens
:class:`StatusHistoryPopup`, which lists them newest first with their time
(warnings and errors in their colors), so a notice that vanished while you
looked elsewhere ("Retuned while recording...") can still be read.
Warnings and errors also go to Tools > Error History (the window logs them).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, List, NamedTuple, Optional

try:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - no Qt
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import get_palette, set_role

#: How many messages are kept.
CAPACITY = 20


class StatusEntry(NamedTuple):
    """One message: when it was (last) shown, its text, tone and how many
    times in a row it was shown."""

    when: float
    text: str
    tone: str
    count: int = 1

    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.when))


class StatusHistory:
    """The last :data:`CAPACITY` status messages (Qt-free)."""

    def __init__(self, capacity: int = CAPACITY):
        self._entries: Deque[StatusEntry] = deque(maxlen=max(1, int(capacity)))

    def add(self, text: str, tone: Optional[str] = None, when: Optional[float] = None):
        """Remember ``text``; the same text shown again in a row counts
        once more instead of filling the list."""
        text = str(text or "").strip()
        if not text:
            return
        stamp = time.time() if when is None else float(when)
        tone = tone or ""
        if self._entries and self._entries[-1].text == text:
            last = self._entries.pop()
            self._entries.append(StatusEntry(stamp, text, tone, last.count + 1))
            return
        self._entries.append(StatusEntry(stamp, text, tone))

    def entries(self) -> List[StatusEntry]:
        """Newest first."""
        return list(reversed(self._entries))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def entry_text(entry: StatusEntry) -> str:
    """``"12:34:56  Retuned to 100.100 MHz (x2)"``."""
    repeat = f"  (×{entry.count})" if entry.count > 1 else ""
    return f"{entry.clock()}  {entry.text}{repeat}"


class StatusHistoryPopup(QFrame if HAS_PYQT6 else object):
    """A popup listing a :class:`StatusHistory`, newest first, with Copy
    and a way on to Tools > Error History."""

    def __init__(
        self,
        history: StatusHistory,
        parent: Any = None,
        open_error_history: Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("statusHistoryPopup")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._history = history
        self._open_errors = open_error_history

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        title = QLabel("RECENT MESSAGES")
        set_role(title, "caption")
        layout.addWidget(title)

        self.list = QListWidget()
        self.list.setAccessibleName("Recent status messages")
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.list, 1)

        self.empty = QLabel(
            "No messages yet. What the app tells you in the status bar is "
            "kept here (the last 20)."
        )
        self.empty.setWordWrap(True)
        set_role(self.empty, "placeholder")
        layout.addWidget(self.empty)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Warnings and errors are also in Tools > Error History.")
        hint.setWordWrap(True)
        set_role(hint, "hint")
        buttons.addWidget(hint, 1)
        self.copy_button = QPushButton("&Copy")
        self.copy_button.setToolTip("Copy the messages (the selected ones, or all)")
        self.copy_button.clicked.connect(self.copy)
        buttons.addWidget(self.copy_button)
        if open_error_history is not None:
            self.errors_button = QPushButton("&Error History...")
            self.errors_button.setToolTip("Tools > Error History (Ctrl+E)")
            self.errors_button.clicked.connect(self._show_errors)
            buttons.addWidget(self.errors_button)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        """Fill the list from the history (newest first)."""
        p = get_palette()
        self.list.clear()
        entries = self._history.entries()
        for entry in entries:
            item = QListWidgetItem(entry_text(entry))
            item.setToolTip(entry.text)
            if entry.tone in ("warning", "danger", "success", "info"):
                item.setForeground(p.qcolor(entry.tone))
            self.list.addItem(item)
        self.list.setVisible(bool(entries))
        self.empty.setVisible(not entries)
        self.copy_button.setEnabled(bool(entries))

    def texts(self) -> List[str]:
        """The listed lines (for tests and Copy)."""
        return [self.list.item(i).text() for i in range(self.list.count())]

    def copy(self) -> None:
        rows = sorted(index.row() for index in self.list.selectedIndexes())
        lines = [self.list.item(r).text() for r in rows] or self.texts()
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _show_errors(self) -> None:
        self.close()
        if self._open_errors is not None:
            self._open_errors()

    def show_above(self, anchor: Any) -> None:
        """Show over the status bar, its bottom-left corner at ``anchor``'s
        top-left, sized to the window."""
        window = anchor.window()
        width = max(360, min(640, window.width() - 40))
        self.resize(width, 300)
        corner = anchor.mapToGlobal(QPoint(0, 0))
        left = max(window.mapToGlobal(QPoint(0, 0)).x() + 8, corner.x())
        self.move(left, corner.y() - self.height() - 4)
        self.show()
        self.list.setFocus(Qt.FocusReason.PopupFocusReason)

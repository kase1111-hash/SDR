"""
The status bar's recent messages.

Status-bar messages clear themselves after a few seconds. The window keeps
the last 20 in a :class:`StatusHistory`; a click on the message area opens
:class:`StatusHistoryPopup`, which lists them newest first with their time
(warnings and errors in their colors), so a notice that vanished while you
looked elsewhere ("Retuned while recording...") can still be read.
Warnings and errors also go to Tools > Error History (the window logs them).

A message too long for its place in the status bar (it keeps 220 px at
1024 px with every chip shown) ends there in "…"; :class:`MessageOverflow`
shows the whole of it just above the status bar while it lasts.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, List, NamedTuple, Optional

try:
    from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer
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
    from .themes import get_palette, set_role, set_tone

#: How many messages are kept.
CAPACITY = 20

#: The whole message's panel: its gap to the status bar and the least room
#: it keeps from the window's edges (px), and the least width it wraps a
#: message in, should the message area sit far to the right.
_OVERFLOW_GAP = 4
_OVERFLOW_EDGE = 8
_OVERFLOW_MIN_WIDTH = 360


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
        top-left, sized to the window (and moved left as far as it takes to
        stay inside it)."""
        window = anchor.window()
        width = max(360, min(640, window.width() - 40))
        self.resize(width, 300)
        corner = anchor.mapToGlobal(QPoint(0, 0))
        window_left = window.mapToGlobal(QPoint(0, 0)).x()
        right = window_left + window.width() - 8  # (the window's right margin)
        left = max(window_left + 8, min(corner.x(), right - width))
        self.move(left, corner.y() - self.height() - 4)
        self.show()
        self.list.setFocus(Qt.FocusReason.PopupFocusReason)


class MessageOverflow(QLabel if HAS_PYQT6 else object):
    """The status bar's message in full, just above the status bar, while
    a message too long for its place there shows (at 1024 px with REC,
    AUDIO and the chips shown the message area keeps 220 px: "Stop
    recording to change the sam…").

    The status bar keeps its layout (the message still ends in "…" there)
    and nothing else moves: this is a panel of the window laid over the
    bottom of the plots and the right column, starting where the message
    starts, wrapped to the window's width, in the message's tone. It shows
    for as long as the message does; a click on it, or Esc in the window,
    hides it until the next message, and so does a click on the message
    area (the last 20 messages take its place). Being part of the window,
    it is never over a dialog or another window. A message that fits shows
    no panel: the status bar looks as it always did.

    ``anchor`` is the status bar's message label: its ``text()`` is the
    whole message and its ``is_elided()`` says whether it is drawn cut
    short. The panel follows it as the window, the status bar and its
    chips change size.
    """

    def __init__(self, window: Any, anchor: Any):
        super().__init__(window)
        self._anchor = anchor
        self._message = ""
        self._dismissed = False
        self.setObjectName("statusMessageOverflow")
        set_role(self, "status-overflow")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setIndent(0)  # (a framed label indents its text by an "x")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Status message")
        self.setToolTip(
            "The status bar's message in full. Click (or press Esc) to hide "
            f"it; click the message in the status bar for the last {CAPACITY}."
        )
        self.hide()
        # Placed again once a resize has been dealt with (the message area
        # cuts its text anew in its own resize event, after this one's).
        self._place_timer = QTimer(self)
        self._place_timer.setSingleShot(True)
        self._place_timer.setInterval(0)
        self._place_timer.timeout.connect(self.sync)
        anchor.installEventFilter(self)
        bar = anchor.parentWidget()
        if bar is not None and bar is not window:
            bar.installEventFilter(self)  # (the window's height moves it)

    def message(self) -> str:
        """The message shown (or held back), "" when none."""
        return self._message

    def show_message(self, text: str, tone: Optional[str] = None) -> None:
        """The status bar now shows ``text`` in ``tone``: shown whole here
        if the status bar cuts it. The same text again keeps a panel the
        user hid hidden (it is still the message they hid)."""
        text = str(text or "")
        if text != self._message:
            self._dismissed = False
        self._message = text
        self.setText(text)
        self.setAccessibleDescription(text)
        set_tone(self, tone)
        self.sync()

    def clear(self) -> None:
        """The message is gone (its time is up)."""
        self._message = ""
        self._dismissed = False
        self.setText("")
        self.setAccessibleDescription("")
        self.hide()

    def dismiss(self) -> None:
        """Hide it until the next message (a click, Esc)."""
        if self._message:
            self._dismissed = True
        self.hide()

    def wanted(self) -> bool:
        """Whether the message is on and the status bar cuts it short."""
        anchor = self._anchor
        elided = getattr(anchor, "is_elided", None)
        return bool(
            self._message
            and not self._dismissed
            and anchor.isVisible()
            and callable(elided)
            and elided()
        )

    def sync(self) -> None:
        """Show, place or hide it as the message and the status bar are."""
        if not self.wanted():
            self.hide()
            return
        self._place()
        self.show()
        self.raise_()

    def _place(self) -> None:
        """Just above the status bar, its text starting where the
        message's text starts: as wide as the message (at least the message
        area), wrapped to what the window has room for from there."""
        window = self.parentWidget()
        anchor = self._anchor
        bar = anchor.parentWidget() or anchor
        self.ensurePolished()
        margins = self.contentsMargins()
        # (Where the status bar draws the message's first letter.)
        start = anchor.mapTo(window, QPoint(anchor.contentsRect().left(), 0))
        start.setX(start.x() - margins.left())
        bar_top = bar.mapTo(window, QPoint(0, 0)).y()
        metrics = self.fontMetrics()
        widest = max(
            (metrics.horizontalAdvance(line) for line in self._message.splitlines()),
            default=0,
        )
        # (+2: a line exactly as wide as the room might wrap its last word.)
        wanted = widest + margins.left() + margins.right() + 2
        width_room = max(1, window.width() - 2 * _OVERFLOW_EDGE)
        from_start = window.width() - _OVERFLOW_EDGE - start.x()
        room = min(width_room, max(from_start, _OVERFLOW_MIN_WIDTH))
        width = min(max(wanted, anchor.width()), room)
        right = window.width() - _OVERFLOW_EDGE - width
        left = max(_OVERFLOW_EDGE, min(start.x(), right))
        height = self.heightForWidth(width)
        if height <= 0:  # pragma: no cover - a word-wrapped label has one
            height = self.sizeHint().height()
        self.setGeometry(left, bar_top - _OVERFLOW_GAP - height, width, height)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt naming
        if event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.Show,
            QEvent.Type.Hide,
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ) and (self._message or self.isVisible()):
            self._place_timer.start()
        return False

    def changeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().changeEvent(event)
        # (A theme switch restyles it: its size may change.)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            if self._message:
                self._place_timer.start()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        event.accept()  # (not on to the window beneath)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismiss()
        event.accept()

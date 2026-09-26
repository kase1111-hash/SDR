"""
Error history viewer.

Captures warning/error-level log records and surfaces them in a dialog so
users can review transient status-bar messages they may have missed.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QHeaderView,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import get_palette, set_role, set_tone, theme_notifier

# One captured record: (created timestamp, level number, logger name, text).
HistoryEntry = Tuple[float, int, str, str]


class _HistoryHandler(logging.Handler):
    """Ring-buffer log handler that keeps the last N records."""

    def __init__(self, capacity: int = 500):
        super().__init__(level=logging.WARNING)
        self._buffer: Deque[HistoryEntry] = deque(maxlen=capacity)
        self._version = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if record.exc_info:
                text += "\n" + logging.Formatter().formatException(record.exc_info)
            elif record.exc_text:
                text += "\n" + record.exc_text
        except Exception:
            self.handleError(record)
            return
        self._buffer.append((record.created, record.levelno, record.name, text))
        self._version += 1

    @property
    def version(self) -> int:
        """Increments whenever a record is added or the history is cleared."""
        return self._version

    def entries(self) -> List[HistoryEntry]:
        """Captured records, oldest first."""
        return list(self._buffer)

    def snapshot(self):
        """Captured records as ``(HH:MM:SS, levelno, "logger: message")``."""
        return [
            (
                datetime.fromtimestamp(created).strftime("%H:%M:%S"),
                lvl,
                f"{name}: {text}",
            )
            for created, lvl, name, text in self._buffer
        ]

    def clear_history(self) -> None:
        self._buffer.clear()
        self._version += 1


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


def _level_name(levelno: int) -> str:
    if levelno >= logging.CRITICAL:
        return "Critical"
    if levelno >= logging.ERROR:
        return "Error"
    return "Warning"


def _short_source(name: str) -> str:
    """``sdr_module.gui.main_window`` -> ``gui.main_window``."""
    prefix = "sdr_module."
    return name[len(prefix) :] if name.startswith(prefix) else name


def group_entries(entries: List[HistoryEntry]) -> List[dict]:
    """Collapse repeats of the same message; newest first.

    Each group is a dict with ``first``, ``last`` (timestamps), ``level``,
    ``source``, ``text`` and ``count``.
    """
    groups: Dict[Tuple[int, str, str], dict] = {}
    for created, lvl, name, text in entries:
        key = (lvl, name, text)
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "first": created,
                "last": created,
                "level": lvl,
                "source": name,
                "text": text,
                "count": 1,
            }
        else:
            group["last"] = max(group["last"], created)
            group["count"] += 1
    return sorted(groups.values(), key=lambda g: g["last"], reverse=True)


def format_group(group: dict) -> str:
    """One group as a plain-text log line (for copy and export)."""
    stamp = datetime.fromtimestamp(group["last"]).strftime("%Y-%m-%d %H:%M:%S")
    repeat = f" (x{group['count']})" if group["count"] > 1 else ""
    return (
        f"[{stamp}] {_level_name(group['level']).upper()} "
        f"{group['source']}: {group['text']}{repeat}"
    )


class _ViewportResizeWatcher(QObject if HAS_PYQT6 else object):
    """Keeps an overlay label covering a table's viewport."""

    def __init__(self, viewport, overlay):
        super().__init__(viewport)
        self._overlay = overlay
        viewport.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._overlay.setGeometry(obj.rect())
        return False


class ErrorLogDialog(QDialog if HAS_PYQT6 else object):
    """Viewer for recent warning/error messages."""

    _COLUMNS = ("Time", "Level", "Source", "Message", "Count")

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self.setWindowTitle("Error History")
        self.resize(760, 480)
        self.setMinimumSize(520, 340)

        self._groups: List[dict] = []
        self._shown_version = -1

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(list(self._COLUMNS))
        header = self._table.horizontalHeader()
        for col in (0, 1, 2, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        # Centered: a right-aligned "×3" would touch the table's border.
        self._table.horizontalHeaderItem(4).setTextAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self._table.horizontalHeaderItem(4).setToolTip(
            "How many times the message was logged"
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Arrow keys pick an entry; Tab moves on to the details and buttons
        # instead of walking every cell (a keyboard trap).
        self._table.setTabKeyNavigation(False)
        self._table.setAccessibleName("Logged warnings and errors")
        self._table.itemSelectionChanged.connect(self._show_details)

        self._details = QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setPlaceholderText("Select an entry to see the full message.")
        self._details.setAccessibleName("Message details")
        set_role(self._details, "terminal")

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._table)
        self._splitter.addWidget(self._details)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([300, 90])
        layout.addWidget(self._splitter, 1)

        # Empty-state overlay on the table
        self._empty = QLabel(self._table.viewport())
        self._empty.setText(
            "No warnings or errors so far.\n"
            "Problems the app reports will be listed here as they happen."
        )
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        set_role(self._empty, "placeholder")
        self._empty_watcher = _ViewportResizeWatcher(
            self._table.viewport(), self._empty
        )

        self._summary = QLabel()
        set_role(self._summary, "hint")
        layout.addWidget(self._summary)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        close_btn = btns.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setDefault(True)

        self._copy_btn = QPushButton("&Copy All")
        self._copy_btn.setAutoDefault(False)
        self._copy_btn.clicked.connect(self._copy_all)
        btns.addButton(self._copy_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self._export_btn = QPushButton("&Export...")
        self._export_btn.setAutoDefault(False)
        self._export_btn.clicked.connect(self._export)
        btns.addButton(self._export_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self._clear_btn = QPushButton("C&lear")
        self._clear_btn.setAutoDefault(False)
        self._clear_btn.setToolTip("Forget all captured warnings and errors")
        self._clear_btn.clicked.connect(self._clear)
        btns.addButton(self._clear_btn, QDialogButtonBox.ButtonRole.ResetRole)
        layout.addWidget(btns)

        # Pick up new records while the dialog is open.
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._refresh_if_changed)
        self._poll.start()
        self._feedback_timer = QTimer(self)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.timeout.connect(self._update_summary)

        theme_notifier().theme_changed.connect(self._on_theme_changed)
        self._refresh()

    # ------------------------------------------------------------------ #
    def _refresh_if_changed(self) -> None:
        if _HISTORY.version != self._shown_version:
            self._refresh()

    def _on_theme_changed(self, _name: str) -> None:
        self._refresh()

    def _refresh(self):
        selected = self._selected_group()
        selected_key = (
            (selected["level"], selected["source"], selected["text"])
            if selected
            else None
        )
        self._shown_version = _HISTORY.version
        self._groups = group_entries(_HISTORY.entries())

        p = get_palette()
        tones = {
            "Critical": p.qcolor("danger"),
            "Error": p.qcolor("danger"),
            "Warning": p.qcolor("warning"),
        }
        self._table.setRowCount(len(self._groups))
        restore_row = -1
        for row, group in enumerate(self._groups):
            level = _level_name(group["level"])
            first_line = group["text"].splitlines()[0] if group["text"] else ""
            last = datetime.fromtimestamp(group["last"])
            time_item = QTableWidgetItem(last.strftime("%H:%M:%S"))
            time_item.setToolTip(last.strftime("%Y-%m-%d %H:%M:%S"))
            level_item = QTableWidgetItem(level)
            level_item.setForeground(tones[level])
            font = level_item.font()
            font.setBold(True)
            level_item.setFont(font)
            source_item = QTableWidgetItem(_short_source(group["source"]))
            source_item.setToolTip(group["source"])
            message_item = QTableWidgetItem(first_line)
            message_item.setToolTip(group["text"])
            count = group["count"]
            count_item = QTableWidgetItem(f"×{count}" if count > 1 else "")
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            for col, item in enumerate(
                (time_item, level_item, source_item, message_item, count_item)
            ):
                self._table.setItem(row, col, item)
            if selected_key == (group["level"], group["source"], group["text"]):
                restore_row = row

        if restore_row >= 0:
            self._table.selectRow(restore_row)
            # Re-selecting the same row emits no selection change, but the
            # group's count and last time may have moved on.
            self._show_details()
        else:
            self._table.clearSelection()
            self._details.clear()

        empty = not self._groups
        self._empty.setVisible(empty)
        self._empty.setGeometry(self._table.viewport().rect())
        self._details.setVisible(not empty)
        for button, tip in (
            (self._copy_btn, "Copy every entry to the clipboard"),
            (self._export_btn, "Save every entry to a text file"),
            (self._clear_btn, "Forget all captured warnings and errors"),
        ):
            button.setEnabled(not empty)
            button.setToolTip(tip if not empty else "Nothing has been logged yet")
        if not self._feedback_timer.isActive():
            self._update_summary()

    def _update_summary(self) -> None:
        if not self._groups:
            self._summary.setText("Warnings and errors from this session appear here.")
        else:
            total = sum(g["count"] for g in self._groups)
            noun = "message" if total == 1 else "messages"
            self._summary.setText(
                f"{total} {noun} this session, newest first. "
                "Repeated messages are grouped."
            )
        set_tone(self._summary, None)

    def _flash(self, text: str, tone: Optional[str] = "success") -> None:
        """Show short feedback in the summary line, then restore it."""
        self._summary.setText(text)
        set_tone(self._summary, tone)
        self._feedback_timer.start(4000)

    def _selected_group(self) -> Optional[dict]:
        rows = self._table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self._groups):
            return None
        return self._groups[rows[0].row()]

    def _show_details(self) -> None:
        group = self._selected_group()
        if group is None:
            self._details.clear()
            return
        first = datetime.fromtimestamp(group["first"]).strftime("%H:%M:%S")
        last = datetime.fromtimestamp(group["last"]).strftime("%H:%M:%S")
        when = (
            f"{last}"
            if group["count"] == 1
            else (f"{group['count']} times, {first} to {last}")
        )
        text = (
            f"{_level_name(group['level'])} from {group['source']} ({when})\n\n"
            f"{group['text']}"
        )
        # Leave the pane alone when nothing changed, so a text selection the
        # user is making survives the once-a-second refresh.
        if text != self._details.toPlainText():
            self._details.setPlainText(text)

    def _all_text(self) -> str:
        return "\n".join(format_group(g) for g in reversed(self._groups))

    def _copy_all(self) -> None:
        if not self._groups:
            return
        QGuiApplication.clipboard().setText(self._all_text())
        self._flash(f"Copied {len(self._groups)} entries to the clipboard.")

    def _export(self) -> None:
        if not self._groups:
            return
        default = f"sdr-error-history-{datetime.now():%Y%m%d-%H%M%S}.txt"
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Error History",
            default,
            "Text files (*.txt);;Log files (*.log);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._all_text() + "\n")
        except OSError as e:
            self._flash(f"Could not save: {e}", "danger")
            return
        self._flash(f"Saved to {path}")

    def _clear(self):
        _HISTORY.clear_history()
        self._refresh()
        self._flash("History cleared.")

    def done(self, result: int) -> None:
        self._poll.stop()
        try:
            theme_notifier().theme_changed.disconnect(self._on_theme_changed)
        except (TypeError, RuntimeError):
            pass
        super().done(result)

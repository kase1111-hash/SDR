"""
Protocol decoder output panel.

Shows what the live protocol decoder produces: a message table, a
plain-text log that is easy to copy, and running statistics.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QBrush,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QKeySequence,
        QPalette,
        QTextBlockFormat,
        QTextCharFormat,
        QTextCursor,
        QTextLayout,
        QTextOption,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMenu,
        QMessageBox,
        QScrollArea,
        QSizePolicy,
        QStyledItemDelegate,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .themes import get_palette, set_role, set_tone

#: Selector entry that runs no decoder. The main window maps any name it
#: does not know (this one included) to "no live decoder".
PROTOCOL_OFF = "Off"

#: Protocol name -> (what it carries, where to listen). The names are the
#: ones the main window maps to decoder implementations; keep them in step.
PROTOCOL_INFO: Dict[str, Tuple[str, str]] = {
    "POCSAG": (
        "Pager messages (512/1200/2400 baud)",
        "around 152–159 MHz and 929–932 MHz",
    ),
    "FLEX": ("Motorola FLEX pager messages", "around 929–932 MHz"),
    "AX.25/APRS": (
        "Packet radio and APRS position reports",
        "144.390 MHz (N. America) or 144.800 MHz (Europe)",
    ),
    "ADS-B": ("Aircraft identity, altitude and position", "1090 MHz"),
    "ACARS": ("Aircraft data-link text messages (AM)", "129–137 MHz"),
    "RDS": ("Station name and radio text from FM broadcasts", "88–108 MHz"),
}

# Table columns.
COL_TIME, COL_PROTOCOL, COL_ADDRESS, COL_MESSAGE = range(4)
_HEADERS = ("Time", "Protocol", "Address", "Message")
# Widest a size-to-contents column may grow before it elides.
_MAX_COL_WIDTH = {COL_PROTOCOL: 110, COL_ADDRESS: 170}
# Share of the table width the Message column keeps in a narrow panel: the
# Address column gives way first (down to _MIN_ADDRESS_WIDTH, eliding).
_MESSAGE_SHARE = 0.4
_MIN_MESSAGE_WIDTH = 120
_MIN_ADDRESS_WIDTH = 64
# Horizontal padding around cell text and (bold, padded) header text.
_CELL_PAD = 14
_HEADER_PAD = 20
# Lines of the selected message shown under the table before it elides.
_DETAIL_LINES = 3

# Item data role flagging a row whose message failed its integrity checks.
_INVALID_ROLE = Qt.ItemDataRole.UserRole.value + 1 if HAS_PYQT6 else 0


class ViewPlaceholder(QObject if HAS_PYQT6 else object):
    """Centered empty-state text laid over an item view's viewport.

    The label uses the ``placeholder`` role, lets clicks through and follows
    the viewport size. Call :meth:`set_text` and :meth:`set_visible` to
    update it.
    """

    def __init__(self, view: Any, text: str = ""):
        super().__init__(view)
        viewport = view.viewport()
        self.label = QLabel(text, viewport)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        set_role(self.label, "placeholder")
        self.label.setGeometry(viewport.rect())
        viewport.installEventFilter(self)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.Resize:
            self.label.setGeometry(obj.rect())
        return False

    def set_text(self, text: str) -> None:
        self.label.setText(text)

    def set_visible(self, visible: bool) -> None:
        self.label.setVisible(visible)

    def is_visible(self) -> bool:
        return not self.label.isHidden()

    def text(self) -> str:
        return self.label.text()


class _MessageDelegate(QStyledItemDelegate if HAS_PYQT6 else object):
    """Tints rows that failed their checks, using the active theme colors."""

    def initStyleOption(self, option: Any, index: Any) -> None:  # noqa: N802
        super().initStyleOption(option, index)
        if index.data(_INVALID_ROLE):
            p = get_palette()
            option.backgroundBrush = QBrush(p.qcolor("danger_bg"))
            option.palette.setColor(QPalette.ColorRole.Text, p.qcolor("danger"))


class _MessageTable(QTableWidget if HAS_PYQT6 else object):
    """Message table: Ctrl+C copies whole rows, Esc clears the selection and
    the columns are refitted whenever the visible width changes."""

    def __init__(self, panel: "DecoderPanel"):
        super().__init__(0, len(_HEADERS))
        self._panel = panel
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        if obj is self.viewport() and event.type() == QEvent.Type.Resize:
            self._panel._fit_columns()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        if event.matches(QKeySequence.StandardKey.Copy):
            self._panel.copy_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.selectionModel().hasSelection():
            self.clearSelection()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        menu = QMenu(self)
        copy_act = menu.addAction("Copy")
        copy_act.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy_act.setEnabled(bool(self.selectionModel().selectedRows()))
        select_act = menu.addAction("Select All")
        select_act.setEnabled(self.rowCount() > 0)
        chosen = menu.exec(event.globalPos())
        if chosen is copy_act:
            self._panel.copy_selected()
        elif chosen is select_act:
            self.selectAll()


def elide_lines(text: str, font: Any, width: int, max_lines: int) -> str:
    """Wrap ``text`` at ``width`` px and cut it to ``max_lines`` lines.

    Returns the text unchanged when it fits; otherwise the kept lines are
    joined with newlines and the last one ends in an ellipsis, so a
    word-wrapped label shows exactly ``max_lines`` lines.
    """
    width = max(int(width), 1)
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    spans: List[Tuple[int, int]] = []
    layout.beginLayout()
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        spans.append((line.textStart(), line.textLength()))
    layout.endLayout()
    if len(spans) <= max_lines:
        return text
    kept = [text[start : start + length].rstrip() for start, length in spans]
    rest = " ".join(text[spans[max_lines - 1][0] :].split())
    last = QFontMetrics(font).elidedText(rest, Qt.TextElideMode.ElideRight, width)
    return "\n".join(kept[: max_lines - 1] + [last])


class _MessageDetail(QFrame if HAS_PYQT6 else object):
    """Card under the table showing the selected message in full.

    Long messages are cut to a few lines (with an ellipsis) so the card
    never crowds out the table; the Log tab always has the complete text.
    """

    def __init__(self, parent: Any = None):
        super().__init__(parent)
        set_role(self, "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        self.meta = QLabel()
        set_role(self.meta, "hint")
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body = QLabel()
        self.body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # Both wrap and never ask for more width than the panel has. Decoded
        # text is data, never markup ("<b>" in a page stays literal).
        for label in (self.meta, self.body):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(label)
        self._full = ""
        self.hide()

    def show_text(self, meta: str, body: str, tone: Optional[str] = None) -> None:
        """Show ``body`` under a ``meta`` line (``tone`` colors the meta)."""
        self.meta.setText(meta)
        set_tone(self.meta, tone)
        self._full = body
        self.body.setVisible(bool(body))
        self._render()
        self.show()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._render()

    def showEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._render()

    def _render(self) -> None:
        # Measure the frame itself: the layout's own geometry is reset
        # whenever it is invalidated (e.g. right after a tone change).
        margins = self.layout().contentsMargins()
        width = self.contentsRect().width() - margins.left() - margins.right()
        shown = elide_lines(self._full, self.body.font(), width, _DETAIL_LINES)
        self.body.setText(shown)
        self.body.setToolTip(self._full if shown != self._full else "")


class TailFollower(QObject if HAS_PYQT6 else object):
    """Keeps a scroll bar pinned to its end until the user scrolls away.

    Views lay out new rows lazily (and not at all while their tab is
    hidden), so instead of scrolling once per insert this re-pins the bar
    whenever its range grows, as long as the user was at the end.
    """

    def __init__(self, bar: Any):
        super().__init__(bar)
        self._bar = bar
        self.following = True
        bar.valueChanged.connect(self._on_value)
        bar.rangeChanged.connect(self._on_range)

    def _on_value(self, value: int) -> None:
        self.following = value >= self._bar.maximum() - 2

    def _on_range(self, _minimum: int, maximum: int) -> None:
        if self.following:
            self._bar.setValue(maximum)

    def reset(self) -> None:
        """Follow the end again (e.g. after the view was cleared)."""
        self.following = True


class DecoderPanel(QWidget if HAS_PYQT6 else object):
    """
    Protocol decoder panel.

    Displays decoded messages and allows protocol selection. The
    ``_enabled_check`` box ("Decode") pauses decoding without losing the
    selected protocol; the main window reads it before feeding samples.
    """

    if HAS_PYQT6:
        protocol_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._messages: List[Dict[str, Any]] = []
        self._max_messages = 1000
        # Totals since the last Clear (the table keeps only the newest rows).
        self._total = 0
        self._invalid = 0
        self._per_protocol: Counter = Counter()
        # Widest cell text seen per size-to-contents column, in px.
        self._col_content: Dict[int, int] = {}

        self._setup_ui()
        self._update_state()
        self._update_stats()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Protocol selector and the decode on/off switch, side by side.
        proto_row = QHBoxLayout()
        proto_row.setSpacing(8)
        proto_label = QLabel("Protocol:")
        proto_row.addWidget(proto_label)

        self._proto_combo = QComboBox()
        self._proto_combo.addItem(PROTOCOL_OFF)
        self._proto_combo.setItemData(
            0,
            "No decoder. Choose a protocol to start decoding.",
            Qt.ItemDataRole.ToolTipRole,
        )
        for name, (what, where) in PROTOCOL_INFO.items():
            self._proto_combo.addItem(name)
            self._proto_combo.setItemData(
                self._proto_combo.count() - 1,
                f"{what}. Usually {where}.",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._proto_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._proto_combo.setMaximumWidth(260)
        self._proto_combo.setAccessibleName("Decoder protocol")
        proto_label.setBuddy(self._proto_combo)
        self._proto_combo.currentTextChanged.connect(self._on_protocol_changed)
        proto_row.addWidget(self._proto_combo, 1)

        self._enabled_check = QCheckBox("Decode")
        self._enabled_check.setChecked(True)
        self._enabled_check.setAccessibleName("Decode messages")
        self._enabled_check.toggled.connect(self._update_state)
        proto_row.addWidget(self._enabled_check)
        proto_row.addStretch(0)
        layout.addLayout(proto_row)

        # Output tabs, with Clear / Export in the tab bar's corner so the
        # table keeps as much height as possible in the short right column.
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._table = _MessageTable(self)
        self._table.setHorizontalHeaderLabels(list(_HEADERS))
        self._table.setItemDelegate(_MessageDelegate(self._table))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(
            self._table.fontMetrics().height() + 10
        )
        header = self._table.horizontalHeader()
        header.setHighlightSections(False)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setMinimumSectionSize(40)
        for col in (COL_TIME, COL_PROTOCOL, COL_ADDRESS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_MESSAGE, QHeaderView.ResizeMode.Stretch)
        self._table.setToolTip(
            "Decoded messages, newest at the bottom. Click a message to read "
            "it in full below; invalid messages (failed checks) are tinted "
            "red. Ctrl+C copies the selected rows."
        )
        self._table.setMinimumHeight(110)
        # Stick to the newest message unless the user scrolled up to read.
        self._table_tail = TailFollower(self._table.verticalScrollBar())
        self._empty = ViewPlaceholder(self._table)
        self._table.itemSelectionChanged.connect(self._update_detail)

        messages_page = QWidget()
        page_layout = QVBoxLayout(messages_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(4)
        page_layout.addWidget(self._table, 1)
        self._detail = _MessageDetail()
        page_layout.addWidget(self._detail)
        self._tabs.addTab(messages_page, "Messages")
        self._tabs.setTabToolTip(0, "Decoded messages as a table")
        self._fit_columns()

        # Plain-text log of every message, wrapped so nothing is cut off.
        self._raw_output = QTextEdit()
        self._raw_output.setReadOnly(True)
        self._raw_output.setAcceptRichText(False)
        self._raw_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._raw_output.document().setMaximumBlockCount(self._max_messages)
        set_role(self._raw_output, "terminal")
        self._log_empty = ViewPlaceholder(
            self._raw_output,
            "Every decoded message is also listed here in full as plain "
            "text, easy to select and copy.",
        )
        self._log_tail = TailFollower(self._raw_output.verticalScrollBar())
        self._tabs.addTab(self._raw_output, "Log")
        self._tabs.setTabToolTip(1, "Plain-text log of decoded messages, easy to copy")

        self._tabs.addTab(self._build_stats(), "Stats")
        self._tabs.setTabToolTip(2, "Message counts since the last Clear")

        corner = QWidget()
        corner_row = QHBoxLayout(corner)
        corner_row.setContentsMargins(0, 0, 0, 2)
        corner_row.setSpacing(4)
        self._clear_btn = QToolButton()
        self._clear_btn.setText("Clear")
        self._clear_btn.clicked.connect(self.clear)
        corner_row.addWidget(self._clear_btn)
        self._export_btn = QToolButton()
        self._export_btn.setText("Export...")
        self._export_btn.clicked.connect(lambda _checked=False: self._export_messages())
        corner_row.addWidget(self._export_btn)
        self._tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        layout.addWidget(self._tabs, 1)

    def _build_stats(self) -> "QWidget":
        """Statistics page: totals plus a per-protocol breakdown."""
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(12, 10, 12, 10)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        def value_label() -> "QLabel":
            label = QLabel()
            set_role(label, "value")
            label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return label

        self._msg_count_label = value_label()
        form.addRow("Messages received", self._msg_count_label)
        self._valid_count_label = value_label()
        form.addRow("Valid", self._valid_count_label)
        self._invalid_count_label = value_label()
        invalid_caption = QLabel("Invalid")
        invalid_caption.setToolTip("Messages that failed a checksum or parity check")
        form.addRow(invalid_caption, self._invalid_count_label)
        self._last_msg_label = value_label()
        form.addRow("Last message", self._last_msg_label)

        by_proto = QLabel("BY PROTOCOL")
        set_role(by_proto, "caption")
        by_proto.setContentsMargins(0, 8, 0, 0)
        form.addRow(by_proto)
        self._proto_none = QLabel("No messages yet.")
        set_role(self._proto_none, "hint")
        form.addRow(self._proto_none)
        # One "name  count" row per protocol seen, added as they appear.
        self._stats_form = form
        self._value_label = value_label
        self._proto_rows: Dict[str, Any] = {}

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(page)
        return area

    def _header_width(self, col: int) -> int:
        """Width that shows a column's (bold, padded) header text in full."""
        bold = QFont(self._table.horizontalHeader().font())
        bold.setBold(True)
        return QFontMetrics(bold).horizontalAdvance(_HEADERS[col]) + _HEADER_PAD

    def _fit_columns(self) -> None:
        """Size Time/Protocol/Address to their contents and let Message
        stretch over the rest.

        In a narrow panel the Address column gives way (eliding, with the
        full text in its tooltip) so Message keeps at least
        ``_MESSAGE_SHARE`` of the width. Fonts are measured now rather than
        at construction, so the widths follow the applied stylesheet.
        """
        table = getattr(self, "_table", None)
        if table is None:
            return
        fm = table.fontMetrics()
        rows = table.verticalHeader()
        if rows.defaultSectionSize() != fm.height() + 10:
            rows.setDefaultSectionSize(fm.height() + 10)
        want: Dict[int, int] = {}
        for col in (COL_TIME, COL_PROTOCOL, COL_ADDRESS):
            if table.isColumnHidden(col):
                continue
            content = self._col_content.get(col, 0)
            if col == COL_TIME:
                content = fm.horizontalAdvance("00:00:00") + _CELL_PAD
            want[col] = max(self._header_width(col), content)

        avail = table.viewport().width()
        message = max(_MIN_MESSAGE_WIDTH, int(avail * _MESSAGE_SHARE))
        overflow = sum(want.values()) + message - avail
        if overflow > 0 and COL_ADDRESS in want:
            floor = min(
                want[COL_ADDRESS],
                max(_MIN_ADDRESS_WIDTH, self._header_width(COL_ADDRESS)),
            )
            want[COL_ADDRESS] = max(floor, want[COL_ADDRESS] - overflow)

        for col, width in want.items():
            if table.columnWidth(col) != width:
                table.setColumnWidth(col, width)

    def _grow_columns(self, texts: Dict[int, str]) -> None:
        """Note new cell text; refit when a column's contents got wider."""
        fm = self._table.fontMetrics()
        grown = False
        for col, text in texts.items():
            wanted = min(fm.horizontalAdvance(text) + _CELL_PAD, _MAX_COL_WIDTH[col])
            if wanted > self._col_content.get(col, 0):
                self._col_content[col] = wanted
                grown = True
        if grown:
            self._fit_columns()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def protocol(self) -> Optional[str]:
        """Selected protocol name, or ``None`` when decoding is off."""
        text = self._proto_combo.currentText()
        return None if text == PROTOCOL_OFF else text

    def is_decoding(self) -> bool:
        """True when a protocol is selected and Decode is ticked."""
        return self.protocol() is not None and self._enabled_check.isChecked()

    def _on_protocol_changed(self, text: str):
        """Handle protocol change."""
        self._update_state()
        self.protocol_changed.emit(text)

    def _update_state(self, *_args: Any) -> None:
        """Refresh tooltips, enabled states and the empty-table hint."""
        proto = self.protocol()
        self._enabled_check.setEnabled(proto is not None)
        if proto is None:
            self._proto_combo.setToolTip(
                "Protocol to decode. Off: no decoder runs. Hover an entry "
                "for where to listen."
            )
            self._enabled_check.setToolTip("Choose a protocol first")
            empty = (
                "No decoder selected.\n"
                "Choose a protocol above, tune to its frequency and start "
                "acquisition. Decoded messages appear here."
            )
        else:
            what, where = PROTOCOL_INFO.get(proto, ("", ""))
            self._proto_combo.setToolTip(
                f"{proto}: {what}. Usually {where}." if what else proto
            )
            self._enabled_check.setToolTip(
                "Untick to pause decoding without changing the protocol"
            )
            if self._enabled_check.isChecked():
                empty = f"Waiting for {proto} messages.\n" + (
                    f"Tune to {where}, then start acquisition."
                    if where
                    else "Start acquisition to decode."
                )
            else:
                empty = "Decoding is paused.\nTick Decode to resume."
        self._empty.set_text(empty)
        self._empty.set_visible(self._table.rowCount() == 0)
        self._update_buttons()

    def _update_buttons(self) -> None:
        has = bool(self._messages)
        self._clear_btn.setEnabled(has)
        self._export_btn.setEnabled(has)
        self._clear_btn.setToolTip(
            "Remove all decoded messages and reset the statistics"
            if has
            else "Nothing to clear yet"
        )
        self._export_btn.setToolTip(
            "Save the decoded messages to a CSV file"
            if has
            else "No messages to export yet"
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        protocol: str,
        address: str,
        content: str,
        valid: bool = True,
        raw: str = "",
    ):
        """
        Add a decoded message.

        Args:
            protocol: Protocol name
            address: Address/ID
            content: Message content
            valid: Whether message was valid
            raw: Raw hex data
        """
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
        protocol, address, content = str(protocol), str(address), str(content)
        valid = bool(valid)

        msg = {
            "time": timestamp,
            "date": now.strftime("%Y-%m-%d"),
            "protocol": protocol,
            "address": address,
            "content": content,
            "valid": valid,
            "raw": raw,
        }
        self._messages.append(msg)
        self._total += 1
        if not valid:
            self._invalid += 1
        self._per_protocol[protocol] += 1

        # Keep the table and the stored list the same length.
        overflow = len(self._messages) - self._max_messages
        if overflow > 0:
            del self._messages[:overflow]
            for _ in range(overflow):
                self._table.removeRow(0)

        row = self._table.rowCount()
        self._table.insertRow(row)
        short_time = timestamp.split(".")[0]
        cells = {
            COL_TIME: (short_time, f"{msg['date']} {timestamp}"),
            COL_PROTOCOL: (protocol, protocol),
            COL_ADDRESS: (address, address),
            COL_MESSAGE: (content or "(empty)", content),
        }
        for col, (text, tip) in cells.items():
            item = QTableWidgetItem(text)
            if not valid:
                item.setData(_INVALID_ROLE, True)
                tip = f"{tip}\nInvalid: failed a checksum or parity check"
            if tip:
                item.setToolTip(tip)
            self._table.setItem(row, col, item)
        self._grow_columns({COL_PROTOCOL: protocol, COL_ADDRESS: address})

        self._append_log(msg)

        self._empty.set_visible(False)
        self._log_empty.set_visible(False)
        self._update_stats()
        self._update_buttons()

    def _append_log(self, msg: Dict[str, Any]) -> None:
        """Append one message to the log with a hanging indent."""
        doc = self._raw_output.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not doc.isEmpty():
            cursor.insertBlock()
        indent = self._raw_output.fontMetrics().horizontalAdvance("0000")
        block = QTextBlockFormat()
        block.setLeftMargin(indent)
        block.setTextIndent(-indent)
        block.setBottomMargin(2)
        cursor.setBlockFormat(block)
        bold = QTextCharFormat()
        bold.setFontWeight(QFont.Weight.Bold)
        plain = QTextCharFormat()
        cursor.insertText(msg["time"], bold)
        text = f"  {msg['protocol']}  {msg['address']}  {msg['content']}"
        if not msg["valid"]:
            text += "  [invalid]"
        if msg["raw"]:
            text += f"  raw: {msg['raw']}"
        cursor.insertText(text, plain)

    def add_adsb_message(
        self,
        icao: str,
        callsign: str = "",
        altitude: int = 0,
        lat: float = 0,
        lon: float = 0,
        speed: float = 0,
    ):
        """Add an ADS-B message with specific formatting."""
        parts = []
        if callsign:
            parts.append(str(callsign).strip())
        if altitude:
            parts.append(f"{altitude:,} ft")
        if lat and lon:
            parts.append(f"{lat:.4f}, {lon:.4f}")
        if speed:
            parts.append(f"{speed:.0f} kt")

        self.add_message("ADS-B", str(icao), "  ·  ".join(parts))

    def add_pocsag_message(self, address: int, content: str, function: int = 0):
        """Add a POCSAG message."""
        addr_str = f"{address} (F{function})"
        self.add_message("POCSAG", addr_str, content)

    def add_aprs_message(
        self, source: str, dest: str, lat: float = 0, lon: float = 0, comment: str = ""
    ):
        """Add an APRS message."""
        parts = []
        if lat and lon:
            parts.append(f"{lat:.4f}, {lon:.4f}")
        if comment:
            parts.append(str(comment).strip())

        self.add_message("APRS", f"{source}>{dest}", "  ·  ".join(parts))

    def _update_stats(self):
        """Update statistics display."""
        total = self._total
        invalid = self._invalid

        self._msg_count_label.setText(f"{total:,}")
        self._valid_count_label.setText(f"{total - invalid:,}")
        self._invalid_count_label.setText(f"{invalid:,}")
        set_tone(self._invalid_count_label, "danger" if invalid else None)

        if self._messages:
            self._last_msg_label.setText(self._messages[-1].get("time", "–"))
        else:
            self._last_msg_label.setText("–")

        if not self._per_protocol and self._proto_rows:
            for label in self._proto_rows.values():
                self._stats_form.removeRow(label)
            self._proto_rows.clear()
        for name, count in self._per_protocol.items():
            label = self._proto_rows.get(name)
            if label is None:
                label = self._value_label()
                self._stats_form.addRow(name, label)
                self._proto_rows[name] = label
            label.setText(f"{count:,}")
        self._proto_none.setVisible(not self._per_protocol)

        # With a single protocol the column only repeats the selector.
        hide_protocol = len(self._per_protocol) <= 1
        if hide_protocol != self._table.isColumnHidden(COL_PROTOCOL):
            self._table.setColumnHidden(COL_PROTOCOL, hide_protocol)
            self._fit_columns()

    def _update_detail(self) -> None:
        """Show the selected message in full under the table."""
        rows = sorted(i.row() for i in self._table.selectionModel().selectedRows())
        if len(rows) > 1:
            self._detail.show_text(
                f"{len(rows):,} messages selected",
                "Ctrl+C copies them as tab-separated text. Esc clears the "
                "selection.",
            )
            return
        if not rows or not 0 <= rows[0] < len(self._messages):
            self._detail.hide()
            return
        msg = self._messages[rows[0]]
        meta = [msg["time"], msg["protocol"]]
        if msg["address"]:
            meta.append(msg["address"])
        if not msg["valid"]:
            meta.append("failed checks")
        self._detail.show_text(
            "  ·  ".join(meta),
            msg["content"] or "(no message text)",
            None if msg["valid"] else "danger",
        )
        # The card takes height from the table; once that layout settles,
        # keep the selected row in view.
        QTimer.singleShot(0, self._ensure_selection_visible)

    def _ensure_selection_visible(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if len(rows) == 1:
            self._table.scrollTo(rows[0])

    def clear(self):
        """Clear all messages."""
        self._messages.clear()
        self._total = 0
        self._invalid = 0
        self._per_protocol.clear()
        self._table.setRowCount(0)
        self._table_tail.reset()
        self._col_content.clear()
        self._fit_columns()
        self._detail.hide()
        self._raw_output.clear()
        self._log_tail.reset()
        self._log_empty.set_visible(True)
        self._update_stats()
        self._update_state()

    def copy_selected(self) -> int:
        """Copy the selected rows to the clipboard as tab-separated text."""
        rows = sorted(i.row() for i in self._table.selectionModel().selectedRows())
        lines = []
        for row in rows:
            if 0 <= row < len(self._messages):
                m = self._messages[row]
                lines.append(
                    "\t".join((m["time"], m["protocol"], m["address"], m["content"]))
                )
        if lines:
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText("\n".join(lines))
        return len(lines)

    def _export_messages(self, path: Optional[str] = None) -> int:
        """
        Export messages to a CSV file.

        Args:
            path: Destination file; a file dialog is shown when omitted.

        Returns:
            Number of messages written (0 if cancelled or on error).
        """
        if not self._messages:
            return 0
        if not path:
            default = datetime.now().strftime("decoded-%Y%m%d-%H%M%S.csv")
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Decoded Messages",
                default,
                "CSV files (*.csv);;All files (*)",
            )
        if not path:
            return 0
        if not os.path.splitext(path)[1]:
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["Date", "Time", "Protocol", "Address", "Content", "Valid", "Raw"]
                )
                for msg in self._messages:
                    writer.writerow(
                        [
                            msg.get("date", ""),
                            msg.get("time", ""),
                            msg.get("protocol", ""),
                            msg.get("address", ""),
                            msg.get("content", ""),
                            msg.get("valid", True),
                            msg.get("raw", ""),
                        ]
                    )
        except OSError as exc:
            QMessageBox.warning(
                self, "Export Failed", f"Could not write the file:\n{exc}"
            )
            return 0

        count = len(self._messages)
        QMessageBox.information(
            self,
            "Export Messages",
            f"Exported {count:,} message{'' if count == 1 else 's'} to:\n{path}",
        )
        return count

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)

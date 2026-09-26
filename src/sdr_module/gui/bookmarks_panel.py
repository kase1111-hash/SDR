"""
Bookmarks / memory channels panel.

A list of saved frequencies the user can tune with a double-click (or
Enter). Stored via GuiSettings; survives restarts.

Channels import and export as CHIRP-compatible CSV (see
`sdr_module.core.chirp_csv`), so memories can be moved between this
application, CHIRP, and a handheld radio.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from PyQt6.QtCore import QEvent, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLineEdit,
        QMenu,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..core.chirp_csv import ChirpCsvError, export_bookmarks_csv, read_chirp_csv
from .decoder_panel import ViewPlaceholder
from .settings_store import GuiSettings

logger = logging.getLogger(__name__)

CSV_FILTER = "CHIRP CSV (*.csv);;All files (*)"

# Tree columns.
COL_NAME, COL_FREQ, COL_MODE = range(3)

_EMPTY_TEXT = (
    "No bookmarks yet.\n"
    "Press Ctrl+B to bookmark the current frequency, or enter one above "
    "and click Add. Use CSV to import a CHIRP channel list."
)


#: Keys every bookmark has. Anything else came from a CHIRP channel.
_BASIC_KEYS = frozenset(("label", "freq_hz"))
#: CHIRP's default mode; imported channels omit it when it is the default.
_CHIRP_DEFAULT_MODE = "FM"


def _num(value: Any, default: float = 0.0) -> float:
    """``float(value)``, or ``default`` for missing or malformed values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _plural(count: int, word: str) -> str:
    """Count with a plural-aware noun, e.g. "1 channel" or "3 channels"."""
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _trim_decimals(whole: str, point: str, frac: str) -> str:
    """Drop trailing zeros but keep at least 3 decimals ("146.520")."""
    return f"{whole}{point}{frac.rstrip('0').ljust(3, '0')}"


def format_mhz(freq_hz: Any) -> str:
    """Format a frequency as MHz with 3 to 6 decimals ("146.520 MHz")."""
    whole, point, frac = f"{_num(freq_hz) / 1e6:.6f}".partition(".")
    return f"{_trim_decimals(whole, point, frac)} MHz"


def _mode(bookmark: Dict[str, Any]) -> str:
    """Mode of a bookmark; imported channels without one use CHIRP's FM."""
    mode = str(bookmark.get("mode") or "")
    if not mode and set(bookmark) - _BASIC_KEYS:
        return _CHIRP_DEFAULT_MODE
    return mode


def _offset_text(bookmark: Dict[str, Any]) -> str:
    """Short repeater-shift text such as "-0.6" (empty for simplex)."""
    duplex = str(bookmark.get("duplex") or "")
    if duplex in ("+", "-"):
        offset = abs(_num(bookmark.get("offset_hz"))) / 1e6
        return f"{duplex}{offset:g}"
    if duplex == "split":
        return "Split"
    if duplex == "off":
        return "RX only"
    return ""


def _details(bookmark: Dict[str, Any]) -> str:
    """Compact mode / tone / shift summary for the Mode column."""
    parts = [p for p in (_mode(bookmark), str(bookmark.get("tone_mode") or "")) if p]
    shift = _offset_text(bookmark)
    if shift:
        parts.append(shift)
    return " ".join(parts)


def _tooltip(bookmark: Dict[str, Any]) -> str:
    """Multi-line description of one bookmark for its tooltip."""
    lines = [
        str(bookmark.get("label", "")) or "(unnamed)",
        format_mhz(bookmark.get("freq_hz")),
    ]
    mode = _mode(bookmark)
    if mode:
        lines.append(f"Mode: {mode}")
    tone_mode = bookmark.get("tone_mode")
    if tone_mode in ("Tone", "TSQL"):
        key = "rtone" if tone_mode == "Tone" else "ctone"
        lines.append(f"{tone_mode}: {_num(bookmark.get(key), 88.5):g} Hz")
    elif tone_mode:
        lines.append(f"Tone mode: {tone_mode}")
    duplex = str(bookmark.get("duplex") or "")
    if duplex in ("+", "-"):
        lines.append(f"Shift: {_offset_text(bookmark)} MHz")
    elif duplex == "split":
        lines.append(f"TX: {format_mhz(bookmark.get('offset_hz'))}")
    elif duplex == "off":
        lines.append("Receive only")
    if bookmark.get("comment"):
        lines.append(str(bookmark["comment"]))
    lines.append("Double-click or press Enter to tune")
    return "\n".join(lines)


class _MhzSpinBox(QDoubleSpinBox if HAS_PYQT6 else object):
    """MHz spin box that accepts 1 Hz steps but shows 3 to 6 decimals, like
    the list ("146.520 MHz", not "146.520000 MHz")."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt API)
        loc = self.locale()
        text = loc.toString(float(value), "f", self.decimals())
        text = text.replace(loc.groupSeparator(), "")
        whole, point, frac = text.partition(loc.decimalPoint())
        return _trim_decimals(whole, point, frac) if point else text


class BookmarksPanel(QWidget if HAS_PYQT6 else object):
    """Panel to manage saved frequency bookmarks."""

    if HAS_PYQT6:
        tune_requested = pyqtSignal(float, str)  # freq_hz, label

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._settings = GuiSettings()
        self._bookmarks: List[Dict[str, Any]] = self._settings.get_bookmarks()
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Add row: name, frequency (MHz), Add.
        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Name (optional)")
        self._label_input.setClearButtonEnabled(True)
        self._label_input.setToolTip(
            "Name for the new bookmark (optional; the frequency is used "
            "when left empty). Press Enter to add."
        )
        self._label_input.setAccessibleName("Bookmark name")
        self._label_input.setMinimumWidth(80)
        self._label_input.returnPressed.connect(self._add_current)
        add_row.addWidget(self._label_input, 1)

        self._freq_input = _MhzSpinBox()
        self._freq_input.setRange(0.001, 6000.0)
        self._freq_input.setDecimals(6)
        self._freq_input.setValue(100.0)
        self._freq_input.setSuffix(" MHz")
        self._freq_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._freq_input.setAccelerated(True)
        self._freq_input.setToolTip("Frequency of the new bookmark, in MHz")
        self._freq_input.setAccessibleName("Bookmark frequency")
        self._freq_input.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        spin_edit = self._freq_input.lineEdit()
        if spin_edit is not None:
            spin_edit.returnPressed.connect(self._add_current)
        add_row.addWidget(self._freq_input)

        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Save this name and frequency as a bookmark (Enter)")
        self._add_btn.setAutoDefault(False)
        self._add_btn.clicked.connect(self._add_current)
        add_row.addWidget(self._add_btn)
        layout.addLayout(add_row)

        # Bookmark list: name | frequency | mode.
        self._list = QTreeWidget()
        self._list.setColumnCount(3)
        self._list.setHeaderLabels(["Name", "Frequency", "Mode"])
        self._list.setRootIsDecorated(False)
        self._list.setUniformRowHeights(True)
        self._list.setAlternatingRowColors(True)
        self._list.setAllColumnsShowFocus(True)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setMinimumHeight(96)
        header = self._list.header()
        header.setStretchLastSection(False)
        header.setHighlightSections(False)
        header.setSectionsMovable(False)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_FREQ, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_MODE, QHeaderView.ResizeMode.ResizeToContents)
        header_item = self._list.headerItem()
        if header_item is not None:
            header_item.setTextAlignment(
                COL_FREQ, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        self._list.itemDoubleClicked.connect(self._on_item_activated)
        self._list.itemSelectionChanged.connect(self._update_buttons)
        self._list.installEventFilter(self)
        self._empty = ViewPlaceholder(self._list, _EMPTY_TEXT)
        layout.addWidget(self._list, 1)

        # Actions on the selection, plus CHIRP CSV import/export.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._tune_btn = QPushButton("Tune")
        self._tune_btn.clicked.connect(self._tune_selected)
        btn_row.addWidget(self._tune_btn)

        self._rename_btn = QPushButton("Rename...")
        self._rename_btn.clicked.connect(self._rename_selected)
        btn_row.addWidget(self._rename_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._remove_btn)
        for btn in (self._tune_btn, self._rename_btn, self._remove_btn):
            btn.setAutoDefault(False)

        btn_row.addStretch(1)

        self._csv_menu = QMenu(self)
        self._import_action = self._csv_menu.addAction("Import CSV...")
        self._import_action.setToolTip("Import memory channels from a CHIRP CSV file")
        self._import_action.triggered.connect(lambda _checked=False: self.import_csv())
        self._export_action = self._csv_menu.addAction("Export CSV...")
        self._export_action.setToolTip(
            "Export saved channels as a CHIRP-compatible CSV file"
        )
        self._export_action.triggered.connect(lambda _checked=False: self.export_csv())
        self._csv_menu.setToolTipsVisible(True)

        self._csv_btn = QPushButton("CSV")
        self._csv_btn.setMenu(self._csv_menu)
        self._csv_btn.setAutoDefault(False)
        self._csv_btn.setToolTip(
            "Import or export channels as a CHIRP-compatible CSV file, to move "
            "memories between this app, CHIRP and a handheld radio"
        )
        btn_row.addWidget(self._csv_btn)
        layout.addLayout(btn_row)

    # ---- Public API ----
    def add_bookmark(self, label: str, freq_hz: float) -> None:
        """Public helper so other parts of the UI can add bookmarks."""
        self._bookmarks.append({"label": label, "freq_hz": float(freq_hz)})
        self._settings.set_bookmarks(self._bookmarks)
        self._refresh_list(select=len(self._bookmarks) - 1)

    def bookmark_count(self) -> int:
        """Number of saved bookmarks."""
        return len(self._bookmarks)

    # ---- CHIRP CSV import / export ----
    def import_csv(self, path: Optional[str] = None) -> int:
        """
        Load memory channels from a CHIRP CSV file.

        Prompts for the file (when `path` is omitted) and for whether to
        replace the existing channels or append to them.

        Args:
            path: CSV file to read; a file dialog is shown when omitted.

        Returns:
            Number of channels imported (0 if cancelled or on error).
        """
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import Channels (CHIRP CSV)", "", CSV_FILTER
            )
        if not path:
            return 0

        try:
            report = read_chirp_csv(path, strict=False)
        except ChirpCsvError as exc:
            logger.warning("Channel import failed: %s", exc)
            QMessageBox.warning(self, "Import Failed", str(exc))
            return 0

        if not report.channels:
            QMessageBox.information(
                self, "Import Channels", "No channels found in that file."
            )
            return 0

        imported = [c.to_bookmark() for c in report.channels]

        if self._bookmarks:
            choice = self._ask_replace_or_append(len(imported))
            if choice is None:
                return 0
            if choice == "replace":
                self._bookmarks = imported
            else:
                self._bookmarks = self._bookmarks + imported
        else:
            self._bookmarks = imported

        self._settings.set_bookmarks(self._bookmarks)
        self._refresh_list()

        if report.skipped:
            QMessageBox.information(
                self,
                "Import Channels",
                f"Imported {_plural(len(imported), 'channel')}.\n"
                f"Skipped {_plural(len(report.skipped), 'unreadable row')}:\n"
                + "\n".join(report.skipped[:10]),
            )
        return len(imported)

    def _ask_replace_or_append(self, count: int) -> Optional[str]:
        """Ask how to merge imported channels: "replace", "append" or None."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Import Channels")
        box.setText(f"Import {_plural(count, 'channel')}.")
        box.setInformativeText(
            f"Replace the existing {_plural(len(self._bookmarks), 'channel')}, "
            "or add the new ones after them?"
        )
        replace_btn = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
        append_btn = box.addButton("Append", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(append_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is replace_btn:
            return "replace"
        if clicked is append_btn:
            return "append"
        return None

    def export_csv(self, path: Optional[str] = None) -> int:
        """
        Save the current channels to a CHIRP-compatible CSV file.

        Args:
            path: Destination file; a file dialog is shown when omitted.

        Returns:
            Number of channels written (0 if cancelled or on error).
        """
        if not self._bookmarks:
            QMessageBox.information(
                self, "Export Channels", "There are no saved channels to export."
            )
            return 0

        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Channels (CHIRP CSV)", "channels.csv", CSV_FILTER
            )
        if not path:
            return 0
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            count = export_bookmarks_csv(path, self._bookmarks)
        except OSError as exc:
            logger.warning("Channel export failed: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))
            return 0

        QMessageBox.information(
            self, "Export Channels", f"Exported {_plural(count, 'channel')} to:\n{path}"
        )
        return count

    def set_current_frequency(self, freq_hz: float) -> None:
        """Update the 'add new' frequency to match the main tuner."""
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(freq_hz / 1e6)
        self._freq_input.blockSignals(False)

    # ---- Internal ----
    def _refresh_list(self, select: Optional[int] = None):
        """Rebuild the rows from ``self._bookmarks`` (optionally selecting one)."""
        self._list.clear()
        any_details = False
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for idx, b in enumerate(self._bookmarks):
            freq = float(b.get("freq_hz", 0.0) or 0.0)
            label = str(b.get("label", "")) or "(unnamed)"
            details = _details(b)
            any_details = any_details or bool(details)
            item = QTreeWidgetItem([label, format_mhz(freq), details])
            item.setTextAlignment(COL_FREQ, right)
            item.setData(COL_NAME, Qt.ItemDataRole.UserRole, idx)
            tip = _tooltip(b)
            for col in (COL_NAME, COL_FREQ, COL_MODE):
                item.setToolTip(col, tip)
            self._list.addTopLevelItem(item)
        # Imported CHIRP channels carry mode/tone detail; hide the column
        # when no bookmark has any, so names get the width.
        self._list.setColumnHidden(COL_MODE, not any_details)
        if select is not None and 0 <= select < self._list.topLevelItemCount():
            item = self._list.topLevelItem(select)
            self._list.setCurrentItem(item)
            self._list.scrollToItem(item)
        self._empty.set_visible(not self._bookmarks)
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Enable the selection actions only when a bookmark is selected."""
        has_sel = self._selected_index() >= 0
        for btn, tip in (
            (self._tune_btn, "Tune the receiver to the selected bookmark (Enter)"),
            (self._rename_btn, "Rename the selected bookmark (F2)"),
            (self._remove_btn, "Delete the selected bookmark (Delete)"),
        ):
            btn.setEnabled(has_sel)
            btn.setToolTip(tip if has_sel else "Select a bookmark in the list first")
        has_any = bool(self._bookmarks)
        self._export_action.setEnabled(has_any)
        self._export_action.setToolTip(
            "Export saved channels as a CHIRP-compatible CSV file"
            if has_any
            else "There are no saved channels to export yet"
        )

    def _selected_index(self) -> int:
        items = self._list.selectedItems()
        if not items:
            return -1
        idx = items[0].data(COL_NAME, Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and 0 <= idx < len(self._bookmarks):
            return idx
        return -1

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        """List keys: Enter tunes, Delete/Backspace removes, F2 renames."""
        if obj is self._list and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._tune_selected()
                return True
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._remove_selected()
                return True
            if key == Qt.Key.Key_F2:
                self._rename_selected()
                return True
        return super().eventFilter(obj, event)

    def _add_current(self):
        freq_hz = self._freq_input.value() * 1e6
        label = self._label_input.text().strip() or format_mhz(freq_hz)
        self.add_bookmark(label, freq_hz)
        self._label_input.clear()

    def _tune_index(self, idx: int) -> None:
        if not 0 <= idx < len(self._bookmarks):
            return
        b = self._bookmarks[idx]
        self.tune_requested.emit(float(b.get("freq_hz", 0.0)), str(b.get("label", "")))

    def _tune_selected(self):
        self._tune_index(self._selected_index())

    def _on_item_activated(self, item: "QTreeWidgetItem", _column: int = 0):
        idx = item.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if isinstance(idx, int):
            self._tune_index(idx)

    def _rename_selected(self):
        idx = self._selected_index()
        if idx < 0:
            return
        current = str(self._bookmarks[idx].get("label", ""))
        new_label, ok = QInputDialog.getText(
            self, "Rename Bookmark", "Name:", text=current
        )
        if ok and new_label.strip():
            self._bookmarks[idx]["label"] = new_label.strip()
            self._settings.set_bookmarks(self._bookmarks)
            self._refresh_list(select=idx)

    def _confirm_remove(self, bookmark: Dict[str, Any]) -> bool:
        """Ask before deleting a bookmark (there is no undo)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Remove Bookmark")
        name = str(bookmark.get("label", "")) or "(unnamed)"
        freq = format_mhz(float(bookmark.get("freq_hz", 0.0) or 0.0))
        box.setText(f"Remove “{name}” ({freq})?")
        box.setInformativeText("This can't be undone.")
        remove_btn = box.addButton("Remove", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(remove_btn)
        box.exec()
        return box.clickedButton() is remove_btn

    def _remove_selected(self):
        idx = self._selected_index()
        if idx < 0:
            return
        if not self._confirm_remove(self._bookmarks[idx]):
            return
        del self._bookmarks[idx]
        self._settings.set_bookmarks(self._bookmarks)
        # Keep a selection nearby so repeated Delete presses keep working.
        self._refresh_list(
            select=min(idx, len(self._bookmarks) - 1) if self._bookmarks else None
        )

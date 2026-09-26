"""
Bookmarks / memory channels panel.

A list of saved frequencies the user can tune with a double-click (or
Enter). Stored via GuiSettings; survives restarts.

A bookmark remembers the demodulation mode it was saved with (and CHIRP
channels carry theirs), so tuning one also restores how to listen to it.

Channels import and export as CHIRP-compatible CSV (see
`sdr_module.core.chirp_csv`), so memories can be moved between this
application, CHIRP, and a handheld radio.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

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

from ..core.chirp_csv import (
    DEFAULT_DEMOD,
    ChirpCsvError,
    chirp_mode_to_demod,
    export_bookmarks_csv,
    read_chirp_csv,
)
from .control_panel import DEMOD_MODES, MAX_FREQUENCY_HZ, MIN_FREQUENCY_HZ
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


#: Keys a bookmark saved in this app has: every bookmark has a label and a
#: frequency; "demod" (a Mode list name) and "fm_deviation" (an FM deviation
#: choice such as "75 kHz") record how it was being listened to. Any other
#: key came from a CHIRP channel.
_BASIC_KEYS = frozenset(("label", "freq_hz", "demod", "fm_deviation"))
#: CHIRP's default mode; imported channels omit it when it is the default.
_CHIRP_DEFAULT_MODE = "FM"
#: Demodulator names of the control panel's Mode list.
_DEMOD_NAMES = frozenset(name for name, _tip in DEMOD_MODES)
#: The Mode list's "no audio" entry, which has no CHIRP equivalent, and
#: how it shows in the Mode column.
_IQ_DEMOD = "None (I/Q)"
_IQ_MODE_TEXT = "I/Q"
#: FM deviation choices for CHIRP's broadcast and narrowband FM modes.
_WFM_DEVIATION = "75 kHz"
_NFM_DEVIATION = "5 kHz"


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
    if mode:
        return mode
    if bookmark.get("demod") == _IQ_DEMOD:
        return _IQ_MODE_TEXT
    if set(bookmark) - _BASIC_KEYS:
        return _CHIRP_DEFAULT_MODE
    return ""


def _chirp_mode(demod: str, fm_deviation: str = "") -> str:
    """CHIRP mode for a Mode list name ("" when CHIRP has none, e.g. I/Q)."""
    if demod == "FM":
        return "WFM" if fm_deviation == _WFM_DEVIATION else "FM"
    return demod if demod in ("AM", "USB", "LSB", "CW") else ""


def tune_mode(bookmark: Dict[str, Any]) -> Tuple[str, str]:
    """``(demod, fm_deviation)`` to listen to a bookmark with.

    ``demod`` is a Mode list name ("FM", "AM", "USB", ...) and
    ``fm_deviation`` an FM deviation choice ("5 kHz", "75 kHz"); either is
    empty when the bookmark doesn't say (old bookmarks keep the current
    mode). CHIRP modes are mapped: NFM/FM to FM at 5 kHz, WFM to FM at
    75 kHz, CWR to CW and so on.
    """
    demod = str(bookmark.get("demod") or "")
    if demod in _DEMOD_NAMES:
        deviation = str(bookmark.get("fm_deviation") or "") if demod == "FM" else ""
        return demod, deviation
    chirp = _mode(bookmark)
    if not chirp or chirp == _IQ_MODE_TEXT:
        return "", ""
    demod = chirp_mode_to_demod(chirp)
    if demod == DEFAULT_DEMOD or demod not in _DEMOD_NAMES:
        return "", ""  # e.g. RTTY or Auto: no matching demodulator here
    if demod == "FM":
        wide = chirp.strip().upper() == "WFM"
        return demod, _WFM_DEVIATION if wide else _NFM_DEVIATION
    return demod, ""


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
        # freq_hz, label, demod mode and FM deviation to listen with (see
        # tune_mode(); both "" when the bookmark doesn't say: keep the
        # current mode). Slots may take just (freq_hz, label).
        tune_requested = pyqtSignal(float, str, str, str)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._settings = GuiSettings()
        self._bookmarks: List[Dict[str, Any]] = self._settings.get_bookmarks()
        # The tuned frequency and a callable returning the current
        # (demod, fm_deviation), so "Add" can store how it is listened to.
        self._tuned_hz: Optional[float] = None
        self._mode_source: Optional[Callable[[], Tuple[str, str]]] = None
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
        # Same range as the tuner, so Add never saves a clamped frequency.
        self._freq_input.setRange(MIN_FREQUENCY_HZ / 1e6, MAX_FREQUENCY_HZ / 1e6)
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

        # Named like the File menu's commands (and the dialogs they open).
        self._csv_menu = QMenu(self)
        self._import_action = self._csv_menu.addAction("Import Channels (CHIRP CSV)...")
        self._import_action.setToolTip("Import memory channels from a CHIRP CSV file")
        self._import_action.triggered.connect(lambda _checked=False: self.import_csv())
        self._export_action = self._csv_menu.addAction("Export Channels (CHIRP CSV)...")
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
    def add_bookmark(
        self, label: str, freq_hz: float, mode: str = "", fm_deviation: str = ""
    ) -> None:
        """Save a bookmark (public, so other parts of the UI can add them).

        Args:
            label: Name shown in the list.
            freq_hz: Frequency in Hz.
            mode: Demodulation mode to restore when it is tuned (a Mode list
                name such as "FM" or "USB"); "" leaves the mode alone.
            fm_deviation: FM deviation choice (e.g. "75 kHz"), for FM.
        """
        bookmark: Dict[str, Any] = {"label": label, "freq_hz": float(freq_hz)}
        if mode in _DEMOD_NAMES:
            bookmark["demod"] = mode
            if mode == "FM" and fm_deviation:
                bookmark["fm_deviation"] = str(fm_deviation)
            chirp = _chirp_mode(mode, str(fm_deviation or ""))
            if chirp:
                bookmark["mode"] = chirp  # shown in the list, exported to CHIRP
        self._bookmarks.append(bookmark)
        self._settings.set_bookmarks(self._bookmarks)
        self._refresh_list(select=len(self._bookmarks) - 1)

    def set_mode_source(self, source: Optional[Callable[[], Tuple[str, str]]]) -> None:
        """Set a callable returning the current ``(demod, fm_deviation)``.

        "Add" then saves the listening mode with the bookmark when its
        frequency is the tuned one (see :meth:`set_current_frequency`).
        """
        self._mode_source = source

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
        box.deleteLater()  # it is parented to the panel; don't keep it around
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
        try:
            freq_hz = float(freq_hz)
        except (TypeError, ValueError):
            return
        if not math.isfinite(freq_hz):
            return
        self._tuned_hz = freq_hz
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

    def _current_mode(self, freq_hz: float) -> Tuple[str, str]:
        """How the tuned frequency is listened to, if ``freq_hz`` is it."""
        if (
            self._mode_source is None
            or self._tuned_hz is None
            or abs(freq_hz - self._tuned_hz) >= 0.5
        ):
            return "", ""  # a typed-in frequency: its mode is unknown
        try:
            mode, deviation = self._mode_source()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not read the current mode: %s", exc)
            return "", ""
        return str(mode or ""), str(deviation or "")

    def _add_current(self):
        freq_hz = float(round(self._freq_input.value() * 1e6))
        label = self._label_input.text().strip() or format_mhz(freq_hz)
        self.add_bookmark(label, freq_hz, *self._current_mode(freq_hz))
        self._label_input.clear()

    def _tune_index(self, idx: int) -> None:
        if not 0 <= idx < len(self._bookmarks):
            return
        b = self._bookmarks[idx]
        mode, deviation = tune_mode(b)
        self.tune_requested.emit(
            _num(b.get("freq_hz")), str(b.get("label", "")), mode, deviation
        )

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
        remove_btn = box.addButton("&Remove", QMessageBox.ButtonRole.DestructiveRole)
        # Enter must not confirm an irreversible delete (a stray Delete or
        # Backspace followed by Enter would lose the bookmark); Alt+R does.
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        removed = box.clickedButton() is remove_btn
        box.deleteLater()  # it is parented to the panel; don't keep it around
        return removed

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

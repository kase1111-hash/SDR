"""
Bookmarks / memory channels panel.

A simple list of saved frequencies the user can tune with a click.
Stored via GuiSettings; survives restarts.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QDoubleSpinBox,
        QHBoxLayout,
        QInputDialog,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .settings_store import GuiSettings


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
        layout.setContentsMargins(4, 4, 4, 4)

        # List
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_item_activated)
        layout.addWidget(self._list, 1)

        # Add row: label, freq MHz, add button
        add_row = QHBoxLayout()
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Label")
        add_row.addWidget(self._label_input, 1)

        self._freq_input = QDoubleSpinBox()
        self._freq_input.setRange(0.001, 6000.0)
        self._freq_input.setDecimals(3)
        self._freq_input.setValue(100.0)
        self._freq_input.setSuffix(" MHz")
        add_row.addWidget(self._freq_input)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_current)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        # Action row
        btn_row = QHBoxLayout()
        tune_btn = QPushButton("Tune")
        tune_btn.clicked.connect(self._tune_selected)
        btn_row.addWidget(tune_btn)

        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_selected)
        btn_row.addWidget(rename_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        layout.addLayout(btn_row)

    def add_bookmark(self, label: str, freq_hz: float) -> None:
        """Public helper so other parts of the UI can add bookmarks."""
        self._bookmarks.append({"label": label, "freq_hz": float(freq_hz)})
        self._settings.set_bookmarks(self._bookmarks)
        self._refresh_list()

    def set_current_frequency(self, freq_hz: float) -> None:
        """Update the 'add new' frequency to match the main tuner."""
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(freq_hz / 1e6)
        self._freq_input.blockSignals(False)

    # ---- Internal ----
    def _refresh_list(self):
        self._list.clear()
        for b in self._bookmarks:
            freq = float(b.get("freq_hz", 0.0))
            label = str(b.get("label", ""))
            text = f"{freq/1e6:10.3f} MHz  —  {label}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, (freq, label))
            self._list.addItem(item)

    def _selected_index(self) -> int:
        rows = self._list.selectionModel().selectedRows()
        if not rows:
            return -1
        return rows[0].row()

    def _add_current(self):
        label = self._label_input.text().strip() or "(unnamed)"
        freq_hz = self._freq_input.value() * 1e6
        self.add_bookmark(label, freq_hz)
        self._label_input.clear()

    def _tune_selected(self):
        idx = self._selected_index()
        if idx < 0:
            return
        b = self._bookmarks[idx]
        self.tune_requested.emit(float(b.get("freq_hz", 0.0)), str(b.get("label", "")))

    def _on_item_activated(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            freq_hz, label = data
            self.tune_requested.emit(float(freq_hz), str(label))

    def _rename_selected(self):
        idx = self._selected_index()
        if idx < 0:
            return
        current = str(self._bookmarks[idx].get("label", ""))
        new_label, ok = QInputDialog.getText(
            self, "Rename Bookmark", "Label:", text=current
        )
        if ok and new_label.strip():
            self._bookmarks[idx]["label"] = new_label.strip()
            self._settings.set_bookmarks(self._bookmarks)
            self._refresh_list()

    def _remove_selected(self):
        idx = self._selected_index()
        if idx < 0:
            return
        del self._bookmarks[idx]
        self._settings.set_bookmarks(self._bookmarks)
        self._refresh_list()

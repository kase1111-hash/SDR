"""
Protocol decoder output panel.

Displays decoded messages from various protocols.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QComboBox, QGroupBox, QPushButton,
        QTextEdit, QTableWidget, QTableWidgetItem,
        QHeaderView, QCheckBox, QTabWidget
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QFont
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class DecoderPanel(QWidget if HAS_PYQT6 else object):
    """
    Protocol decoder panel.

    Displays decoded messages and allows protocol selection.
    """

    if HAS_PYQT6:
        protocol_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._messages: List[Dict[str, Any]] = []
        self._max_messages = 1000

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)

        # Protocol selection
        proto_layout = QHBoxLayout()
        proto_layout.addWidget(QLabel("Protocol:"))

        self._proto_combo = QComboBox()
        self._proto_combo.addItems([
            "Auto Detect",
            "POCSAG",
            "FLEX",
            "AX.25/APRS",
            "ADS-B",
            "ACARS",
            "RDS"
        ])
        self._proto_combo.currentTextChanged.connect(self._on_protocol_changed)
        proto_layout.addWidget(self._proto_combo)

        proto_layout.addStretch()

        # Enable checkbox
        self._enabled_check = QCheckBox("Enabled")
        self._enabled_check.setChecked(True)
        proto_layout.addWidget(self._enabled_check)

        layout.addLayout(proto_layout)

        # Tabbed output
        tabs = QTabWidget()

        # Messages table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Time", "Protocol", "Address", "Message"])
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        tabs.addTab(self._table, "Messages")

        # Raw output
        self._raw_output = QTextEdit()
        self._raw_output.setReadOnly(True)
        self._raw_output.setFont(QFont("Monospace", 9))
        tabs.addTab(self._raw_output, "Raw")

        # Statistics
        stats_widget = QWidget()
        stats_layout = QGridLayout(stats_widget)

        stats_layout.addWidget(QLabel("Messages received:"), 0, 0)
        self._msg_count_label = QLabel("0")
        stats_layout.addWidget(self._msg_count_label, 0, 1)

        stats_layout.addWidget(QLabel("Valid:"), 1, 0)
        self._valid_count_label = QLabel("0")
        stats_layout.addWidget(self._valid_count_label, 1, 1)

        stats_layout.addWidget(QLabel("Invalid:"), 2, 0)
        self._invalid_count_label = QLabel("0")
        stats_layout.addWidget(self._invalid_count_label, 2, 1)

        stats_layout.addWidget(QLabel("Last message:"), 3, 0)
        self._last_msg_label = QLabel("-")
        stats_layout.addWidget(self._last_msg_label, 3, 1)

        stats_layout.setRowStretch(4, 1)

        tabs.addTab(stats_widget, "Stats")

        layout.addWidget(tabs)

        # Control buttons
        btn_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        btn_layout.addWidget(clear_btn)

        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._export_messages)
        btn_layout.addWidget(export_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

    def _on_protocol_changed(self, text: str):
        """Handle protocol change."""
        self.protocol_changed.emit(text)

    def add_message(self, protocol: str, address: str, content: str, valid: bool = True, raw: str = ""):
        """
        Add a decoded message.

        Args:
            protocol: Protocol name
            address: Address/ID
            content: Message content
            valid: Whether message was valid
            raw: Raw hex data
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Store message
        msg = {
            "time": timestamp,
            "protocol": protocol,
            "address": address,
            "content": content,
            "valid": valid,
            "raw": raw
        }
        self._messages.append(msg)

        # Trim if too many
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

        # Add to table
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QTableWidgetItem(timestamp))
        self._table.setItem(row, 1, QTableWidgetItem(protocol))
        self._table.setItem(row, 2, QTableWidgetItem(address))
        self._table.setItem(row, 3, QTableWidgetItem(content))

        # Color invalid messages
        if not valid:
            for col in range(4):
                item = self._table.item(row, col)
                if item:
                    item.setBackground(QColor(80, 40, 40))

        # Scroll to bottom
        self._table.scrollToBottom()

        # Add to raw output
        if raw:
            self._raw_output.append(f"[{timestamp}] {protocol}: {raw}")

        # Update stats
        self._update_stats()

    def add_adsb_message(
        self,
        icao: str,
        callsign: str = "",
        altitude: int = 0,
        lat: float = 0,
        lon: float = 0,
        speed: float = 0
    ):
        """Add an ADS-B message with specific formatting."""
        if callsign:
            content = f"{callsign} "
        else:
            content = ""

        if altitude:
            content += f"Alt:{altitude}ft "
        if lat and lon:
            content += f"Pos:{lat:.4f},{lon:.4f} "
        if speed:
            content += f"Spd:{speed:.0f}kt"

        self.add_message("ADS-B", icao, content.strip())

    def add_pocsag_message(self, address: int, content: str, function: int = 0):
        """Add a POCSAG message."""
        addr_str = f"{address} ({function})"
        self.add_message("POCSAG", addr_str, content)

    def add_aprs_message(
        self,
        source: str,
        dest: str,
        lat: float = 0,
        lon: float = 0,
        comment: str = ""
    ):
        """Add an APRS message."""
        content = ""
        if lat and lon:
            content = f"Pos:{lat:.4f},{lon:.4f} "
        content += comment

        self.add_message("APRS", f"{source}>{dest}", content.strip())

    def _update_stats(self):
        """Update statistics display."""
        total = len(self._messages)
        valid = sum(1 for m in self._messages if m.get("valid", True))
        invalid = total - valid

        self._msg_count_label.setText(str(total))
        self._valid_count_label.setText(str(valid))
        self._invalid_count_label.setText(str(invalid))

        if self._messages:
            last = self._messages[-1]
            self._last_msg_label.setText(last.get("time", "-"))

    def clear(self):
        """Clear all messages."""
        self._messages.clear()
        self._table.setRowCount(0)
        self._raw_output.clear()
        self._update_stats()

    def _export_messages(self):
        """Export messages to file."""
        from PyQt6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Messages",
            "", "CSV Files (*.csv);;Text Files (*.txt)"
        )

        if filename:
            with open(filename, "w") as f:
                f.write("Time,Protocol,Address,Content,Valid\n")
                for msg in self._messages:
                    line = (
                        f'"{msg.get("time", "")}","{msg.get("protocol", "")}",'
                        f'"{msg.get("address", "")}","{msg.get("content", "")}",'
                        f'{msg.get("valid", True)}\n'
                    )
                    f.write(line)

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)

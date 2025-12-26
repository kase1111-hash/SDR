"""
NatLangChain GUI Panel for SDR application.

Provides a user interface for:
- Creating and broadcasting blockchain entries
- Viewing received entries from radio
- Monitoring chain synchronization status
- Managing peer connections
- Displaying blockchain state
"""

from __future__ import annotations

from typing import Optional, List, Dict
from datetime import datetime

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QTextEdit, QPushButton, QGroupBox,
        QListWidget, QListWidgetItem, QTabWidget, QSplitter,
        QComboBox, QSpinBox, QCheckBox, QFrame, QScrollArea
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..protocols.natlangchain import (
    NatLangChainRadio, NLCEntry, NLCBlock, NLCMessageType
)


class EntryWidget(QWidget if HAS_PYQT6 else object):
    """Widget displaying a single NatLangChain entry."""

    def __init__(self, entry: NLCEntry, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self.entry = entry
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header: author, timestamp, hash
        header_layout = QHBoxLayout()

        author_label = QLabel(f"📡 {self.entry.author}")
        author_label.setStyleSheet("font-weight: bold; color: #4a9;")
        header_layout.addWidget(author_label)

        timestamp = datetime.fromtimestamp(self.entry.timestamp)
        time_label = QLabel(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        time_label.setStyleSheet("color: #888; font-size: 10px;")
        header_layout.addWidget(time_label)

        header_layout.addStretch()

        hash_label = QLabel(f"#{self.entry.entry_hash}")
        hash_label.setStyleSheet("color: #68a; font-size: 10px; font-family: monospace;")
        header_layout.addWidget(hash_label)

        layout.addLayout(header_layout)

        # Intent
        intent_label = QLabel(f"Intent: {self.entry.intent}")
        intent_label.setStyleSheet("color: #a84; font-style: italic;")
        layout.addWidget(intent_label)

        # Content
        content_text = QLabel(self.entry.content)
        content_text.setWordWrap(True)
        content_text.setStyleSheet("background-color: #222; padding: 8px; border-radius: 4px;")
        layout.addWidget(content_text)

        # Separator
        self.setStyleSheet("border-bottom: 1px solid #333;")


class NatLangChainPanel(QWidget if HAS_PYQT6 else object):
    """
    Main NatLangChain panel for the SDR application.

    Provides interface for blockchain entry creation, broadcasting,
    and monitoring the distributed ledger over radio.
    """

    if HAS_PYQT6:
        entry_broadcast = pyqtSignal(object)  # NLCEntry
        chain_sync_requested = pyqtSignal()
        peer_announce_requested = pyqtSignal()

    def __init__(self, callsign: str = "N0CALL", parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._callsign = callsign
        self._nlc_radio = NatLangChainRadio(
            callsign=callsign,
            on_entry_received=self._on_entry_received,
            on_block_received=self._on_block_received,
            on_peer_discovered=self._on_peer_discovered
        )

        self._entries: List[NLCEntry] = []
        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget
        tabs = QTabWidget()

        # Tab 1: Create & Broadcast
        create_tab = self._create_compose_tab()
        tabs.addTab(create_tab, "📝 Compose")

        # Tab 2: Received Entries
        entries_tab = self._create_entries_tab()
        tabs.addTab(entries_tab, "📥 Entries")

        # Tab 3: Chain Status
        chain_tab = self._create_chain_tab()
        tabs.addTab(chain_tab, "⛓ Chain")

        # Tab 4: Peers
        peers_tab = self._create_peers_tab()
        tabs.addTab(peers_tab, "📡 Peers")

        layout.addWidget(tabs)

    def _create_compose_tab(self) -> QWidget:
        """Create the compose/broadcast tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Callsign setting
        callsign_layout = QHBoxLayout()
        callsign_layout.addWidget(QLabel("Callsign:"))
        self._callsign_input = QLineEdit(self._callsign)
        self._callsign_input.setMaxLength(10)
        self._callsign_input.setFixedWidth(100)
        self._callsign_input.textChanged.connect(self._on_callsign_changed)
        callsign_layout.addWidget(self._callsign_input)
        callsign_layout.addStretch()
        layout.addLayout(callsign_layout)

        # Intent input
        intent_group = QGroupBox("Intent (brief purpose)")
        intent_layout = QVBoxLayout(intent_group)
        self._intent_input = QLineEdit()
        self._intent_input.setPlaceholderText("e.g., 'Offer amateur radio antenna for sale'")
        self._intent_input.setMaxLength(100)
        intent_layout.addWidget(self._intent_input)
        layout.addWidget(intent_group)

        # Content input
        content_group = QGroupBox("Content (natural language prose)")
        content_layout = QVBoxLayout(content_group)
        self._content_input = QTextEdit()
        self._content_input.setPlaceholderText(
            "Write your entry in natural language...\n\n"
            "Examples:\n"
            "• 'I am offering a Yaesu FT-991A for $800, includes original box and manual.'\n"
            "• 'Looking for someone to help install a tower in the Phoenix area.'\n"
            "• 'Completed my first satellite QSO today with SO-50!'"
        )
        self._content_input.setMinimumHeight(150)
        content_layout.addWidget(self._content_input)

        # Character count
        self._char_count = QLabel("0 / 500 characters")
        self._char_count.setStyleSheet("color: #888; font-size: 10px;")
        self._content_input.textChanged.connect(self._update_char_count)
        content_layout.addWidget(self._char_count)

        layout.addWidget(content_group)

        # Broadcast options
        options_layout = QHBoxLayout()

        self._compress_check = QCheckBox("Compress for TX")
        self._compress_check.setChecked(True)
        options_layout.addWidget(self._compress_check)

        self._repeat_spin = QSpinBox()
        self._repeat_spin.setRange(1, 5)
        self._repeat_spin.setValue(1)
        self._repeat_spin.setPrefix("Repeat: ")
        self._repeat_spin.setSuffix("x")
        options_layout.addWidget(self._repeat_spin)

        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Buttons
        button_layout = QHBoxLayout()

        self._preview_btn = QPushButton("👁 Preview")
        self._preview_btn.clicked.connect(self._preview_entry)
        button_layout.addWidget(self._preview_btn)

        self._broadcast_btn = QPushButton("📡 Broadcast Entry")
        self._broadcast_btn.setStyleSheet(
            "background-color: #2a5; color: white; font-weight: bold; padding: 8px;"
        )
        self._broadcast_btn.clicked.connect(self._broadcast_entry)
        button_layout.addWidget(self._broadcast_btn)

        layout.addLayout(button_layout)

        # Status
        self._tx_status = QLabel("Ready to broadcast")
        self._tx_status.setStyleSheet("color: #888;")
        layout.addWidget(self._tx_status)

        layout.addStretch()
        return widget

    def _create_entries_tab(self) -> QWidget:
        """Create the received entries tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Filter bar
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Entries", "My Entries", "Received Only"])
        self._filter_combo.currentIndexChanged.connect(self._filter_entries)
        filter_layout.addWidget(self._filter_combo)

        filter_layout.addStretch()

        self._entry_count = QLabel("0 entries")
        filter_layout.addWidget(self._entry_count)

        layout.addLayout(filter_layout)

        # Entries list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._entries_container = QWidget()
        self._entries_layout = QVBoxLayout(self._entries_container)
        self._entries_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._entries_layout.setSpacing(8)

        scroll.setWidget(self._entries_container)
        layout.addWidget(scroll)

        # Placeholder when empty
        self._no_entries_label = QLabel(
            "No entries received yet.\n\n"
            "• Entries from other stations will appear here\n"
            "• Request chain sync to fetch historical entries\n"
            "• Announce your presence to discover peers"
        )
        self._no_entries_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_entries_label.setStyleSheet("color: #666;")
        self._entries_layout.addWidget(self._no_entries_label)

        return widget

    def _create_chain_tab(self) -> QWidget:
        """Create the blockchain status tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Chain stats
        stats_group = QGroupBox("Blockchain Status")
        stats_layout = QGridLayout(stats_group)

        stats_layout.addWidget(QLabel("Chain Length:"), 0, 0)
        self._chain_length = QLabel("0 blocks")
        self._chain_length.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self._chain_length, 0, 1)

        stats_layout.addWidget(QLabel("Latest Block:"), 1, 0)
        self._latest_hash = QLabel("(genesis)")
        self._latest_hash.setStyleSheet("font-family: monospace; font-size: 10px;")
        stats_layout.addWidget(self._latest_hash, 1, 1)

        stats_layout.addWidget(QLabel("Pending Entries:"), 2, 0)
        self._pending_count = QLabel("0")
        stats_layout.addWidget(self._pending_count, 2, 1)

        stats_layout.addWidget(QLabel("Known Peers:"), 3, 0)
        self._peer_count = QLabel("0")
        stats_layout.addWidget(self._peer_count, 3, 1)

        layout.addWidget(stats_group)

        # Sync controls
        sync_group = QGroupBox("Chain Synchronization")
        sync_layout = QVBoxLayout(sync_group)

        sync_btn_layout = QHBoxLayout()

        self._sync_btn = QPushButton("🔄 Request Sync")
        self._sync_btn.clicked.connect(self._request_sync)
        sync_btn_layout.addWidget(self._sync_btn)

        self._announce_btn = QPushButton("📡 Announce Presence")
        self._announce_btn.clicked.connect(self._announce_presence)
        sync_btn_layout.addWidget(self._announce_btn)

        sync_layout.addLayout(sync_btn_layout)

        self._sync_status = QLabel("Not synced")
        self._sync_status.setStyleSheet("color: #888;")
        sync_layout.addWidget(self._sync_status)

        layout.addWidget(sync_group)

        # Recent blocks
        blocks_group = QGroupBox("Recent Blocks")
        blocks_layout = QVBoxLayout(blocks_group)

        self._blocks_list = QListWidget()
        self._blocks_list.setMaximumHeight(200)
        blocks_layout.addWidget(self._blocks_list)

        layout.addWidget(blocks_group)

        layout.addStretch()
        return widget

    def _create_peers_tab(self) -> QWidget:
        """Create the peers management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Peer list
        peers_group = QGroupBox("Discovered Peers")
        peers_layout = QVBoxLayout(peers_group)

        self._peers_list = QListWidget()
        peers_layout.addWidget(self._peers_list)

        # Peer actions
        peer_btn_layout = QHBoxLayout()

        self._query_peers_btn = QPushButton("🔍 Query Peers")
        self._query_peers_btn.clicked.connect(self._query_peers)
        peer_btn_layout.addWidget(self._query_peers_btn)

        self._direct_sync_btn = QPushButton("⬇ Sync from Selected")
        self._direct_sync_btn.clicked.connect(self._sync_from_peer)
        peer_btn_layout.addWidget(self._direct_sync_btn)

        peers_layout.addLayout(peer_btn_layout)
        layout.addWidget(peers_group)

        # Connection info
        info_group = QGroupBox("Connection Info")
        info_layout = QVBoxLayout(info_group)

        info_text = QLabel(
            "NatLangChain peers are discovered via radio broadcast.\n\n"
            "• Peers announce their presence periodically\n"
            "• Chain data is exchanged via packet radio\n"
            "• All transmissions require valid amateur license"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #888;")
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        layout.addStretch()
        return widget

    def _setup_timers(self):
        """Setup periodic update timers."""
        # Update stats every 5 seconds
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(5000)

        # Cleanup stale fragments every 30 seconds
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(
            self._nlc_radio._assembler.cleanup_stale
        )
        self._cleanup_timer.start(30000)

    def _on_callsign_changed(self, text: str):
        """Handle callsign change."""
        self._callsign = text.upper()
        self._nlc_radio.callsign = self._callsign

    def _update_char_count(self):
        """Update character count display."""
        count = len(self._content_input.toPlainText())
        self._char_count.setText(f"{count} / 500 characters")
        if count > 500:
            self._char_count.setStyleSheet("color: #d44; font-size: 10px;")
        else:
            self._char_count.setStyleSheet("color: #888; font-size: 10px;")

    def _preview_entry(self):
        """Preview the entry before broadcast."""
        content = self._content_input.toPlainText().strip()
        intent = self._intent_input.text().strip()

        if not content or not intent:
            self._tx_status.setText("⚠️ Please enter both intent and content")
            self._tx_status.setStyleSheet("color: #da4;")
            return

        entry = self._nlc_radio.create_entry(content, intent)

        # Show preview
        preview = (
            f"Entry Preview:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Author: {entry.author}\n"
            f"Intent: {entry.intent}\n"
            f"Hash: {entry.entry_hash}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{entry.content}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Compressed size: {len(entry.to_compressed())} bytes"
        )

        self._tx_status.setText(preview)
        self._tx_status.setStyleSheet("color: #4a9;")

    def _broadcast_entry(self):
        """Create and broadcast an entry."""
        content = self._content_input.toPlainText().strip()
        intent = self._intent_input.text().strip()

        if not content or not intent:
            self._tx_status.setText("⚠️ Please enter both intent and content")
            self._tx_status.setStyleSheet("color: #da4;")
            return

        if len(content) > 500:
            self._tx_status.setText("⚠️ Content too long (max 500 chars)")
            self._tx_status.setStyleSheet("color: #da4;")
            return

        entry = self._nlc_radio.create_entry(content, intent)
        packets = self._nlc_radio.broadcast_entry(entry)

        self._entries.append(entry)
        self._add_entry_widget(entry)

        self._tx_status.setText(
            f"✓ Broadcast entry {entry.entry_hash} ({len(packets)} packets)"
        )
        self._tx_status.setStyleSheet("color: #4a9;")

        # Clear inputs
        self._content_input.clear()
        self._intent_input.clear()

        self.entry_broadcast.emit(entry)

    def _on_entry_received(self, entry: NLCEntry):
        """Handle received entry from radio."""
        self._entries.append(entry)
        self._add_entry_widget(entry)
        self._update_stats()

    def _on_block_received(self, block: NLCBlock):
        """Handle received block from radio."""
        self._nlc_radio.add_block_to_chain(block)
        self._update_stats()

        # Add to blocks list
        item = QListWidgetItem(
            f"Block #{block.index} - {len(block.entries)} entries - {block.block_hash[:12]}..."
        )
        self._blocks_list.insertItem(0, item)

    def _on_peer_discovered(self, callsign: str):
        """Handle discovered peer."""
        # Check if already in list
        for i in range(self._peers_list.count()):
            if self._peers_list.item(i).text().startswith(callsign):
                return

        item = QListWidgetItem(f"{callsign} - Active")
        self._peers_list.addItem(item)
        self._update_stats()

    def _add_entry_widget(self, entry: NLCEntry):
        """Add an entry widget to the entries list."""
        # Remove placeholder if present
        if self._no_entries_label.isVisible():
            self._no_entries_label.hide()

        widget = EntryWidget(entry)
        self._entries_layout.insertWidget(0, widget)
        self._entry_count.setText(f"{len(self._entries)} entries")

    def _filter_entries(self, index: int):
        """Filter displayed entries."""
        # TODO: Implement filtering
        pass

    def _request_sync(self):
        """Request chain synchronization."""
        self._nlc_radio.request_chain_sync()
        self._sync_status.setText("Sync requested, waiting for response...")
        self._sync_status.setStyleSheet("color: #4a9;")
        self.chain_sync_requested.emit()

    def _announce_presence(self):
        """Announce presence to peers."""
        self._nlc_radio.announce_presence()
        self._sync_status.setText("Presence announced")
        self._sync_status.setStyleSheet("color: #4a9;")
        self.peer_announce_requested.emit()

    def _query_peers(self):
        """Query for active peers."""
        self._nlc_radio.announce_presence()

    def _sync_from_peer(self):
        """Sync from selected peer."""
        selected = self._peers_list.currentItem()
        if selected:
            callsign = selected.text().split(" - ")[0]
            self._nlc_radio.request_chain_sync(target=callsign)
            self._sync_status.setText(f"Sync requested from {callsign}")

    def _update_stats(self):
        """Update displayed statistics."""
        summary = self._nlc_radio.get_chain_summary()

        self._chain_length.setText(f"{summary['length']} blocks")
        self._latest_hash.setText(summary['latest_hash'] or "(genesis)")
        self._pending_count.setText(str(summary['pending_entries']))
        self._peer_count.setText(str(len(summary['known_peers'])))

    def set_callsign(self, callsign: str):
        """Set the operator callsign."""
        self._callsign = callsign.upper()
        self._callsign_input.setText(self._callsign)
        self._nlc_radio.callsign = self._callsign

    def get_radio_interface(self) -> NatLangChainRadio:
        """Get the underlying radio interface for SDR integration."""
        return self._nlc_radio

    def process_received_data(self, data: bytes):
        """Process data received from SDR."""
        result = self._nlc_radio.receive_packet(data)
        if result:
            self._tx_status.setText(f"Received: {result['type']} from {result['source']}")


__all__ = [
    'NatLangChainPanel',
    'EntryWidget',
]

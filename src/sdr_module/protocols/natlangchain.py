"""
NatLangChain Protocol Integration for SDR Radio Transmission.

This module enables transmission and reception of NatLangChain blockchain
data over amateur radio frequencies. It supports:
- Natural language entry transmission via packet radio (AX.25)
- Chain synchronization between radio-connected nodes
- Proof of Understanding validation over the air
- Compressed and error-corrected data encoding

NatLangChain is a prose-first blockchain where natural language entries
form the primary ledger substrate. This integration allows decentralized
blockchain synchronization without internet connectivity.

Usage:
    from sdr_module.protocols.natlangchain import NatLangChainRadio

    nlc = NatLangChainRadio(callsign="W1ABC")
    nlc.broadcast_entry(content="...", intent="...")
    nlc.request_chain_sync()
"""

from __future__ import annotations

import json
import hashlib
import zlib
import base64
import logging
import time
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NLCMessageType(Enum):
    """NatLangChain radio message types."""
    ENTRY = auto()              # New blockchain entry
    ENTRY_ACK = auto()          # Entry received acknowledgment
    CHAIN_REQUEST = auto()      # Request chain sync
    CHAIN_RESPONSE = auto()     # Chain data response
    BLOCK_ANNOUNCE = auto()     # New block mined announcement
    VALIDATION_REQUEST = auto() # Request entry validation (PoU)
    VALIDATION_RESPONSE = auto()# Validation paraphrase response
    PEER_ANNOUNCE = auto()      # Node presence announcement
    PEER_QUERY = auto()         # Query for active peers
    HEARTBEAT = auto()          # Keep-alive signal


@dataclass
class NLCEntry:
    """
    A NatLangChain natural language entry for radio transmission.

    This is a simplified version of the full NatLangChain entry format,
    optimized for radio transmission with compression.
    """
    content: str                    # The natural language prose
    author: str                     # Callsign or identifier
    intent: str                     # Brief purpose summary
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_hash: str = ""            # SHA-256 hash of content

    def __post_init__(self):
        """Calculate entry hash if not provided."""
        if not self.entry_hash:
            self.entry_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of entry content."""
        data = f"{self.content}|{self.author}|{self.intent}|{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "NLCEntry":
        """Create entry from dictionary."""
        return cls(**data)

    def to_compressed(self) -> bytes:
        """Compress entry for radio transmission."""
        json_data = json.dumps(self.to_dict(), separators=(',', ':'))
        return zlib.compress(json_data.encode(), level=9)

    @classmethod
    def from_compressed(cls, data: bytes) -> "NLCEntry":
        """Decompress entry from radio transmission."""
        json_data = zlib.decompress(data).decode()
        return cls.from_dict(json.loads(json_data))


@dataclass
class NLCBlock:
    """
    A simplified NatLangChain block for radio transmission.
    """
    index: int
    entries: List[NLCEntry]
    timestamp: float
    previous_hash: str
    block_hash: str = ""
    nonce: int = 0

    def __post_init__(self):
        if not self.block_hash:
            self.block_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate block hash."""
        entry_hashes = "|".join(e.entry_hash for e in self.entries)
        data = f"{self.index}|{entry_hashes}|{self.timestamp}|{self.previous_hash}|{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "entries": [e.to_dict() for e in self.entries],
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "block_hash": self.block_hash,
            "nonce": self.nonce
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NLCBlock":
        """Create block from dictionary."""
        entries = [NLCEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            index=data["index"],
            entries=entries,
            timestamp=data["timestamp"],
            previous_hash=data["previous_hash"],
            block_hash=data.get("block_hash", ""),
            nonce=data.get("nonce", 0)
        )


@dataclass
class NLCRadioPacket:
    """
    Radio packet format for NatLangChain transmission.

    Format: [TYPE:1][SEQ:2][TOTAL:2][HASH:8][PAYLOAD:variable]

    Large messages are fragmented into multiple packets with sequence
    numbers for reassembly.
    """
    msg_type: NLCMessageType
    source_callsign: str
    dest_callsign: str          # "CQ" for broadcast
    sequence: int = 0           # Fragment sequence number
    total_fragments: int = 1    # Total fragments in message
    message_hash: str = ""      # Hash for fragment reassembly
    payload: bytes = b""
    timestamp: float = field(default_factory=time.time)

    # Maximum payload size per packet (leaving room for AX.25 overhead)
    MAX_PAYLOAD_SIZE = 200

    def to_bytes(self) -> bytes:
        """Serialize packet for transmission."""
        header = {
            "t": self.msg_type.value,
            "s": self.source_callsign,
            "d": self.dest_callsign,
            "q": self.sequence,
            "n": self.total_fragments,
            "h": self.message_hash,
            "ts": self.timestamp
        }
        header_json = json.dumps(header, separators=(',', ':')).encode()
        header_len = len(header_json).to_bytes(2, 'big')
        return header_len + header_json + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "NLCRadioPacket":
        """Deserialize packet from received data."""
        header_len = int.from_bytes(data[:2], 'big')
        header_json = data[2:2+header_len].decode()
        header = json.loads(header_json)
        payload = data[2+header_len:]

        return cls(
            msg_type=NLCMessageType(header["t"]),
            source_callsign=header["s"],
            dest_callsign=header["d"],
            sequence=header["q"],
            total_fragments=header["n"],
            message_hash=header["h"],
            payload=payload,
            timestamp=header.get("ts", time.time())
        )

    @classmethod
    def fragment_message(
        cls,
        msg_type: NLCMessageType,
        source: str,
        dest: str,
        payload: bytes
    ) -> List["NLCRadioPacket"]:
        """Fragment a large message into multiple packets."""
        message_hash = hashlib.sha256(payload).hexdigest()[:8]

        fragments = []
        total = (len(payload) + cls.MAX_PAYLOAD_SIZE - 1) // cls.MAX_PAYLOAD_SIZE
        total = max(1, total)

        for i in range(total):
            start = i * cls.MAX_PAYLOAD_SIZE
            end = min(start + cls.MAX_PAYLOAD_SIZE, len(payload))
            fragment_payload = payload[start:end]

            fragments.append(cls(
                msg_type=msg_type,
                source_callsign=source,
                dest_callsign=dest,
                sequence=i,
                total_fragments=total,
                message_hash=message_hash,
                payload=fragment_payload
            ))

        return fragments


class FragmentAssembler:
    """Reassembles fragmented radio packets."""

    def __init__(self, timeout_seconds: float = 60.0):
        self._pending: Dict[str, Dict[int, bytes]] = {}
        self._metadata: Dict[str, Dict] = {}
        self._timeout = timeout_seconds

    def add_fragment(self, packet: NLCRadioPacket) -> Optional[bytes]:
        """
        Add a fragment and return complete message if all fragments received.

        Returns:
            Complete payload bytes if message is complete, None otherwise.
        """
        key = f"{packet.source_callsign}:{packet.message_hash}"

        # Initialize storage for this message
        if key not in self._pending:
            self._pending[key] = {}
            self._metadata[key] = {
                "msg_type": packet.msg_type,
                "source": packet.source_callsign,
                "dest": packet.dest_callsign,
                "total": packet.total_fragments,
                "timestamp": time.time()
            }

        # Store fragment
        self._pending[key][packet.sequence] = packet.payload

        # Check if complete
        if len(self._pending[key]) == packet.total_fragments:
            # Reassemble in order
            complete = b"".join(
                self._pending[key][i]
                for i in range(packet.total_fragments)
            )
            # Clean up
            del self._pending[key]
            del self._metadata[key]
            return complete

        return None

    def cleanup_stale(self):
        """Remove incomplete messages that have timed out."""
        now = time.time()
        stale_keys = [
            key for key, meta in self._metadata.items()
            if now - meta["timestamp"] > self._timeout
        ]
        for key in stale_keys:
            del self._pending[key]
            del self._metadata[key]
            logger.debug(f"Cleaned up stale fragments: {key}")


class NatLangChainRadio:
    """
    NatLangChain radio interface for SDR transmission.

    Provides high-level API for transmitting and receiving blockchain
    data over amateur radio frequencies.
    """

    def __init__(
        self,
        callsign: str,
        on_entry_received: Optional[Callable[[NLCEntry], None]] = None,
        on_block_received: Optional[Callable[[NLCBlock], None]] = None,
        on_peer_discovered: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize NatLangChain radio interface.

        Args:
            callsign: Amateur radio callsign for identification
            on_entry_received: Callback for received entries
            on_block_received: Callback for received blocks
            on_peer_discovered: Callback for discovered peers
        """
        self.callsign = callsign.upper()
        self._on_entry = on_entry_received
        self._on_block = on_block_received
        self._on_peer = on_peer_discovered

        self._assembler = FragmentAssembler()
        self._known_peers: Dict[str, float] = {}  # callsign -> last_seen
        self._pending_entries: List[NLCEntry] = []
        self._local_chain: List[NLCBlock] = []
        self._tx_queue: List[NLCRadioPacket] = []

        # Callbacks for SDR integration
        self._tx_callback: Optional[Callable[[bytes], None]] = None

        logger.info(f"NatLangChain Radio initialized for {self.callsign}")

    def set_tx_callback(self, callback: Callable[[bytes], None]):
        """Set callback for transmitting data via SDR."""
        self._tx_callback = callback

    def create_entry(
        self,
        content: str,
        intent: str,
        metadata: Optional[Dict] = None
    ) -> NLCEntry:
        """
        Create a new blockchain entry.

        Args:
            content: Natural language prose content
            intent: Brief purpose summary
            metadata: Optional additional data

        Returns:
            The created NLCEntry
        """
        entry = NLCEntry(
            content=content,
            author=self.callsign,
            intent=intent,
            metadata=metadata or {}
        )
        self._pending_entries.append(entry)
        logger.info(f"Created entry: {entry.entry_hash}")
        return entry

    def broadcast_entry(self, entry: NLCEntry) -> List[NLCRadioPacket]:
        """
        Broadcast an entry to all listening stations.

        Args:
            entry: The entry to broadcast

        Returns:
            List of packets to transmit
        """
        payload = entry.to_compressed()
        packets = NLCRadioPacket.fragment_message(
            NLCMessageType.ENTRY,
            self.callsign,
            "CQ",  # Broadcast
            payload
        )

        self._tx_queue.extend(packets)
        logger.info(f"Queued entry broadcast: {entry.entry_hash} ({len(packets)} packets)")

        # Transmit if callback set
        if self._tx_callback:
            for packet in packets:
                self._tx_callback(packet.to_bytes())

        return packets

    def request_chain_sync(self, target: str = "CQ") -> NLCRadioPacket:
        """
        Request blockchain synchronization from peers.

        Args:
            target: Specific callsign or "CQ" for broadcast

        Returns:
            The request packet
        """
        # Include our latest block hash if we have a chain
        sync_request = {
            "latest_index": len(self._local_chain) - 1 if self._local_chain else -1,
            "latest_hash": self._local_chain[-1].block_hash if self._local_chain else ""
        }
        payload = json.dumps(sync_request).encode()

        packet = NLCRadioPacket(
            msg_type=NLCMessageType.CHAIN_REQUEST,
            source_callsign=self.callsign,
            dest_callsign=target,
            payload=payload
        )

        self._tx_queue.append(packet)

        if self._tx_callback:
            self._tx_callback(packet.to_bytes())

        logger.info(f"Requested chain sync from {target}")
        return packet

    def announce_presence(self) -> NLCRadioPacket:
        """
        Announce presence to discover other NatLangChain nodes.

        Returns:
            The announcement packet
        """
        announce_data = {
            "callsign": self.callsign,
            "chain_length": len(self._local_chain),
            "pending_entries": len(self._pending_entries),
            "version": "1.0"
        }
        payload = json.dumps(announce_data).encode()

        packet = NLCRadioPacket(
            msg_type=NLCMessageType.PEER_ANNOUNCE,
            source_callsign=self.callsign,
            dest_callsign="CQ",
            payload=payload
        )

        if self._tx_callback:
            self._tx_callback(packet.to_bytes())

        logger.info("Announced presence on frequency")
        return packet

    def receive_packet(self, data: bytes) -> Optional[Dict]:
        """
        Process a received radio packet.

        Args:
            data: Raw packet bytes

        Returns:
            Decoded message data if complete, None if fragment pending
        """
        try:
            packet = NLCRadioPacket.from_bytes(data)
        except Exception as e:
            logger.warning(f"Failed to parse packet: {e}")
            return None

        # Skip our own transmissions
        if packet.source_callsign == self.callsign:
            return None

        # Check if addressed to us or broadcast
        if packet.dest_callsign not in (self.callsign, "CQ"):
            return None

        logger.debug(f"Received {packet.msg_type.name} from {packet.source_callsign}")

        # Handle fragmented messages
        if packet.total_fragments > 1:
            complete_payload = self._assembler.add_fragment(packet)
            if complete_payload is None:
                return None
            payload = complete_payload
        else:
            payload = packet.payload

        # Process by message type
        return self._process_message(packet.msg_type, packet.source_callsign, payload)

    def _process_message(
        self,
        msg_type: NLCMessageType,
        source: str,
        payload: bytes
    ) -> Optional[Dict]:
        """Process a complete message."""

        result = {"type": msg_type.name, "source": source}

        if msg_type == NLCMessageType.ENTRY:
            entry = NLCEntry.from_compressed(payload)
            result["entry"] = entry.to_dict()
            logger.info(f"Received entry from {source}: {entry.entry_hash}")
            if self._on_entry:
                self._on_entry(entry)

        elif msg_type == NLCMessageType.PEER_ANNOUNCE:
            peer_data = json.loads(payload.decode())
            self._known_peers[source] = time.time()
            result["peer_data"] = peer_data
            logger.info(f"Discovered peer: {source}")
            if self._on_peer:
                self._on_peer(source)

        elif msg_type == NLCMessageType.CHAIN_REQUEST:
            request_data = json.loads(payload.decode())
            result["request"] = request_data
            # Auto-respond with our chain if we have data they need
            self._respond_to_sync_request(source, request_data)

        elif msg_type == NLCMessageType.CHAIN_RESPONSE:
            chain_data = json.loads(zlib.decompress(payload).decode())
            blocks = [NLCBlock.from_dict(b) for b in chain_data.get("blocks", [])]
            result["blocks"] = [b.to_dict() for b in blocks]
            for block in blocks:
                if self._on_block:
                    self._on_block(block)

        elif msg_type == NLCMessageType.BLOCK_ANNOUNCE:
            block = NLCBlock.from_dict(json.loads(payload.decode()))
            result["block"] = block.to_dict()
            logger.info(f"New block announced: #{block.index}")
            if self._on_block:
                self._on_block(block)

        elif msg_type == NLCMessageType.VALIDATION_REQUEST:
            val_data = json.loads(payload.decode())
            result["validation_request"] = val_data
            # Could trigger local validation here

        return result

    def _respond_to_sync_request(self, requester: str, request: Dict):
        """Respond to a chain sync request."""
        their_index = request.get("latest_index", -1)

        if len(self._local_chain) <= their_index + 1:
            # They're caught up or ahead
            return

        # Send blocks they're missing
        blocks_to_send = self._local_chain[their_index + 1:]

        chain_data = {
            "blocks": [b.to_dict() for b in blocks_to_send],
            "total_length": len(self._local_chain)
        }

        payload = zlib.compress(json.dumps(chain_data).encode())
        packets = NLCRadioPacket.fragment_message(
            NLCMessageType.CHAIN_RESPONSE,
            self.callsign,
            requester,
            payload
        )

        if self._tx_callback:
            for packet in packets:
                self._tx_callback(packet.to_bytes())

        logger.info(f"Sent {len(blocks_to_send)} blocks to {requester}")

    def add_block_to_chain(self, block: NLCBlock) -> bool:
        """
        Add a validated block to the local chain.

        Args:
            block: The block to add

        Returns:
            True if block was added successfully
        """
        # Validate chain linkage
        if self._local_chain:
            if block.previous_hash != self._local_chain[-1].block_hash:
                logger.warning(f"Block {block.index} has invalid previous hash")
                return False
            if block.index != len(self._local_chain):
                logger.warning(f"Block index mismatch: expected {len(self._local_chain)}")
                return False
        elif block.index != 0:
            logger.warning("First block must have index 0")
            return False

        self._local_chain.append(block)
        logger.info(f"Added block #{block.index} to chain")
        return True

    def get_chain_summary(self) -> Dict:
        """Get summary of local blockchain state."""
        return {
            "length": len(self._local_chain),
            "latest_hash": self._local_chain[-1].block_hash if self._local_chain else None,
            "pending_entries": len(self._pending_entries),
            "known_peers": list(self._known_peers.keys()),
            "tx_queue_size": len(self._tx_queue)
        }

    def get_known_peers(self) -> List[str]:
        """Get list of known peer callsigns."""
        # Clean up stale peers (not seen in 10 minutes)
        cutoff = time.time() - 600
        self._known_peers = {
            k: v for k, v in self._known_peers.items()
            if v > cutoff
        }
        return list(self._known_peers.keys())


# Convenience function for creating a radio-ready entry
def create_radio_entry(
    callsign: str,
    content: str,
    intent: str,
    **metadata
) -> NLCEntry:
    """
    Create a NatLangChain entry ready for radio transmission.

    Args:
        callsign: Author's amateur radio callsign
        content: Natural language prose content
        intent: Brief purpose summary
        **metadata: Additional metadata fields

    Returns:
        NLCEntry ready for transmission
    """
    return NLCEntry(
        content=content,
        author=callsign.upper(),
        intent=intent,
        metadata=metadata
    )


__all__ = [
    'NLCMessageType',
    'NLCEntry',
    'NLCBlock',
    'NLCRadioPacket',
    'NatLangChainRadio',
    'FragmentAssembler',
    'create_radio_entry',
]

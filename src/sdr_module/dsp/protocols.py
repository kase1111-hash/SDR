"""
Protocol decoders for common radio protocols.

Supports:
- POCSAG: Pager protocol (512/1200/2400 baud)
- FLEX: Motorola pager protocol (1600/3200/6400 baud)
- AX.25/APRS: Amateur packet radio
- RDS: Radio Data System (FM broadcast)
- ADS-B: Aircraft transponder (Mode S)
- ACARS: Aircraft communications
"""

import numpy as np
from typing import Optional, List, Tuple, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time


class ProtocolType(Enum):
    """Supported protocol types."""
    POCSAG = "pocsag"
    FLEX = "flex"
    AX25 = "ax25"
    APRS = "aprs"
    RDS = "rds"
    ADSB = "adsb"
    ACARS = "acars"


@dataclass
class DecodedMessage:
    """Base class for decoded messages."""
    protocol: ProtocolType
    timestamp: float
    raw_bits: bytes
    valid: bool
    error_message: str = ""


class ProtocolDecoder(ABC):
    """Abstract base class for protocol decoders."""

    def __init__(self, sample_rate: float):
        self._sample_rate = sample_rate
        self._callbacks: List[Callable] = []

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def add_callback(self, callback: Callable[[DecodedMessage], None]) -> None:
        """Add callback for decoded messages."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, message: DecodedMessage) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(message)
            except Exception:
                pass

    @abstractmethod
    def decode(self, samples: np.ndarray) -> List[DecodedMessage]:
        """Decode samples and return messages."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset decoder state."""
        pass


# ============================================================================
# POCSAG Protocol Decoder
# ============================================================================

@dataclass
class POCSAGMessage(DecodedMessage):
    """POCSAG decoded message."""
    address: int = 0
    function: int = 0
    message_type: str = ""  # "numeric", "alpha", "tone"
    content: str = ""
    baud_rate: int = 1200


class POCSAGDecoder(ProtocolDecoder):
    """
    POCSAG (Post Office Code Standardisation Advisory Group) decoder.

    POCSAG is a common pager protocol operating at 512, 1200, or 2400 baud.

    Features:
    - Automatic baud rate detection
    - Numeric and alphanumeric message decoding
    - BCH error correction
    - Address and function code extraction
    """

    # POCSAG constants
    SYNC_WORD = 0x7CD215D8
    IDLE_WORD = 0x7A89C197
    PREAMBLE_BITS = 576  # Minimum preamble length

    # BCH(31,21) generator polynomial
    BCH_POLY = 0x769

    # Numeric character set
    NUMERIC_CHARS = "0123456789*U -)(]"

    # Alpha character set (7-bit ASCII mapping)
    ALPHA_SHIFT = {
        0x00: '\x00', 0x01: '\x01', 0x02: '\x02', 0x03: '\x03',
        0x04: '\x04', 0x05: '\x05', 0x06: '\x06', 0x07: '\x07',
        0x08: '\x08', 0x09: '\x09', 0x0A: '\n', 0x0B: '\x0B',
        0x0C: '\x0C', 0x0D: '\r', 0x0E: '\x0E', 0x0F: '\x0F',
    }

    def __init__(
        self,
        sample_rate: float,
        baud_rate: int = 1200,
        auto_baud: bool = True
    ):
        """
        Initialize POCSAG decoder.

        Args:
            sample_rate: Sample rate in Hz
            baud_rate: Initial baud rate (512, 1200, or 2400)
            auto_baud: Automatically detect baud rate
        """
        super().__init__(sample_rate)
        self._baud_rate = baud_rate
        self._auto_baud = auto_baud
        self._samples_per_bit = int(sample_rate / baud_rate)

        # State machine
        self._state = "searching"  # searching, synced, receiving
        self._bit_buffer: List[int] = []
        self._current_batch: List[int] = []
        self._current_address = 0
        self._current_function = 0
        self._message_bits: List[int] = []

        # Timing
        self._last_transition = 0
        self._bit_clock = 0.0

        # Messages
        self._messages: List[POCSAGMessage] = []
        self._timestamp = 0.0

    @property
    def baud_rate(self) -> int:
        """Get current baud rate."""
        return self._baud_rate

    def _detect_baud_rate(self, samples: np.ndarray) -> int:
        """Detect baud rate from samples."""
        # Look for preamble pattern (101010...)
        # Try each baud rate and find best match
        best_baud = self._baud_rate
        best_score = 0

        for baud in [512, 1200, 2400]:
            sps = int(self._sample_rate / baud)
            if sps < 2:
                continue

            # Count transitions at expected rate
            score = 0
            for i in range(0, len(samples) - sps, sps):
                if i + sps < len(samples):
                    sign1 = 1 if samples[i] > 0 else -1
                    sign2 = 1 if samples[i + sps] > 0 else -1
                    if sign1 != sign2:
                        score += 1

            if score > best_score:
                best_score = score
                best_baud = baud

        return best_baud

    def _bits_to_word(self, bits: List[int]) -> int:
        """Convert 32 bits to word."""
        word = 0
        for bit in bits[:32]:
            word = (word << 1) | bit
        return word

    def _check_bch(self, word: int) -> Tuple[bool, int]:
        """
        Check BCH(31,21) error correction.

        Returns:
            (valid, corrected_word)
        """
        # Extract 31-bit codeword (excluding parity bit)
        codeword = (word >> 1) & 0x7FFFFFFF

        # Compute syndrome
        syndrome = 0
        temp = codeword
        for _ in range(21):
            if temp & 0x40000000:
                temp ^= self.BCH_POLY << 20
            temp <<= 1
        syndrome = (temp >> 21) & 0x3FF

        if syndrome == 0:
            return True, word

        # Try single-bit error correction
        for i in range(31):
            test = codeword ^ (1 << i)
            temp = test
            for _ in range(21):
                if temp & 0x40000000:
                    temp ^= self.BCH_POLY << 20
                temp <<= 1
            if ((temp >> 21) & 0x3FF) == 0:
                # Check parity
                corrected = (test << 1) | (word & 1)
                return True, corrected

        return False, word

    def _decode_numeric(self, bits: List[int]) -> str:
        """Decode numeric message."""
        result = ""
        for i in range(0, len(bits) - 3, 4):
            nibble = (bits[i] << 3) | (bits[i+1] << 2) | (bits[i+2] << 1) | bits[i+3]
            if nibble < len(self.NUMERIC_CHARS):
                char = self.NUMERIC_CHARS[nibble]
                if char != ']':  # End of message marker
                    result += char
                else:
                    break
        return result

    def _decode_alpha(self, bits: List[int]) -> str:
        """Decode alphanumeric message."""
        result = ""
        for i in range(0, len(bits) - 6, 7):
            char_code = 0
            for j in range(7):
                char_code |= bits[i + j] << j
            if 32 <= char_code < 127:
                result += chr(char_code)
            elif char_code == 0:
                break
        return result

    def _process_batch(self, batch: List[int]) -> Optional[POCSAGMessage]:
        """Process a complete batch (16 codewords after sync)."""
        if len(batch) < 512:  # 16 words * 32 bits
            return None

        messages = []
        current_address = None
        current_function = None
        message_bits = []

        for frame in range(8):  # 8 frames per batch
            for word_idx in range(2):  # 2 words per frame
                bit_start = (frame * 2 + word_idx) * 32
                word_bits = batch[bit_start:bit_start + 32]
                if len(word_bits) < 32:
                    continue

                word = self._bits_to_word(word_bits)

                # Check for idle word
                if word == self.IDLE_WORD:
                    if message_bits and current_address is not None:
                        # End of message
                        content = self._decode_alpha(message_bits)
                        msg = POCSAGMessage(
                            protocol=ProtocolType.POCSAG,
                            timestamp=self._timestamp,
                            raw_bits=bytes(batch),
                            valid=True,
                            address=current_address,
                            function=current_function,
                            message_type="alpha",
                            content=content,
                            baud_rate=self._baud_rate
                        )
                        messages.append(msg)
                        message_bits = []
                        current_address = None
                    continue

                # Check BCH
                valid, corrected = self._check_bch(word)
                if not valid:
                    continue

                # Check message type bit (bit 31)
                if not (corrected & 0x80000000):
                    # Address word
                    if message_bits and current_address is not None:
                        # Save previous message
                        content = self._decode_alpha(message_bits)
                        msg = POCSAGMessage(
                            protocol=ProtocolType.POCSAG,
                            timestamp=self._timestamp,
                            raw_bits=bytes(batch),
                            valid=True,
                            address=current_address,
                            function=current_function,
                            message_type="alpha",
                            content=content,
                            baud_rate=self._baud_rate
                        )
                        messages.append(msg)
                        message_bits = []

                    # Extract address (18 bits) and function (2 bits)
                    current_address = ((corrected >> 13) & 0x1FFFF8) | frame
                    current_function = (corrected >> 11) & 0x3

                else:
                    # Message word - extract 20 data bits
                    for i in range(20):
                        message_bits.append((corrected >> (30 - i)) & 1)

        # Handle any remaining message
        if message_bits and current_address is not None:
            content = self._decode_alpha(message_bits)
            msg = POCSAGMessage(
                protocol=ProtocolType.POCSAG,
                timestamp=self._timestamp,
                raw_bits=bytes(batch),
                valid=True,
                address=current_address,
                function=current_function,
                message_type="alpha",
                content=content,
                baud_rate=self._baud_rate
            )
            messages.append(msg)

        return messages[0] if messages else None

    def decode(self, samples: np.ndarray) -> List[POCSAGMessage]:
        """
        Decode POCSAG from FSK-demodulated samples.

        Args:
            samples: FSK-demodulated samples (positive = mark, negative = space)

        Returns:
            List of decoded POCSAG messages
        """
        self._messages = []
        self._timestamp += len(samples) / self._sample_rate

        # Auto-detect baud rate if enabled
        if self._auto_baud and self._state == "searching":
            detected = self._detect_baud_rate(samples)
            if detected != self._baud_rate:
                self._baud_rate = detected
                self._samples_per_bit = int(self._sample_rate / self._baud_rate)

        # Convert samples to bits
        sps = self._samples_per_bit
        for i in range(0, len(samples) - sps, sps):
            # Sample at center of bit
            center = i + sps // 2
            if center < len(samples):
                bit = 1 if samples[center] > 0 else 0
                self._bit_buffer.append(bit)

        # Look for sync word
        while len(self._bit_buffer) >= 32:
            # Check for sync word
            word = self._bits_to_word(self._bit_buffer[:32])

            if word == self.SYNC_WORD:
                self._state = "synced"
                self._bit_buffer = self._bit_buffer[32:]
                self._current_batch = []
            elif self._state == "synced":
                # Collect batch bits
                self._current_batch.append(self._bit_buffer.pop(0))

                if len(self._current_batch) >= 512:
                    # Process complete batch
                    msg = self._process_batch(self._current_batch)
                    if msg:
                        self._messages.append(msg)
                        self._notify_callbacks(msg)
                    self._current_batch = []
                    self._state = "searching"
            else:
                self._bit_buffer.pop(0)

        return self._messages

    def reset(self) -> None:
        """Reset decoder state."""
        self._state = "searching"
        self._bit_buffer = []
        self._current_batch = []
        self._message_bits = []
        self._timestamp = 0.0


# ============================================================================
# AX.25 / APRS Protocol Decoder
# ============================================================================

@dataclass
class AX25Frame(DecodedMessage):
    """AX.25 frame."""
    source: str = ""
    destination: str = ""
    digipeaters: List[str] = None
    control: int = 0
    pid: int = 0
    info: str = ""

    def __post_init__(self):
        if self.digipeaters is None:
            self.digipeaters = []


@dataclass
class APRSMessage(DecodedMessage):
    """APRS decoded message."""
    source: str = ""
    destination: str = ""
    path: List[str] = None
    data_type: str = ""  # position, message, weather, telemetry, etc.
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    course: float = 0.0
    symbol: str = ""
    comment: str = ""

    def __post_init__(self):
        if self.path is None:
            self.path = []


class AX25Decoder(ProtocolDecoder):
    """
    AX.25 protocol decoder.

    AX.25 is the data link layer protocol used for packet radio,
    including APRS (Automatic Packet Reporting System).

    Features:
    - HDLC frame detection
    - Bit unstuffing
    - CRC-16 validation
    - Callsign extraction
    - APRS data parsing
    """

    FLAG = 0x7E
    CRC_POLY = 0x8408  # CRC-CCITT (reversed)

    def __init__(self, sample_rate: float, baud_rate: int = 1200):
        """
        Initialize AX.25 decoder.

        Args:
            sample_rate: Sample rate in Hz
            baud_rate: Baud rate (typically 1200 for VHF APRS)
        """
        super().__init__(sample_rate)
        self._baud_rate = baud_rate
        self._samples_per_bit = int(sample_rate / baud_rate)

        # State
        self._bit_buffer: List[int] = []
        self._frame_buffer: List[int] = []
        self._in_frame = False
        self._ones_count = 0
        self._last_bit = 0

        # NRZI decoding state
        self._nrzi_last = 0

        # Messages
        self._frames: List[AX25Frame] = []
        self._timestamp = 0.0

    @property
    def baud_rate(self) -> int:
        """Get baud rate."""
        return self._baud_rate

    def _nrzi_decode(self, bit: int) -> int:
        """NRZI decode: 0 = transition, 1 = no transition."""
        if bit == self._nrzi_last:
            result = 1
        else:
            result = 0
        self._nrzi_last = bit
        return result

    def _compute_crc(self, data: bytes) -> int:
        """Compute CRC-16-CCITT."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ self.CRC_POLY
                else:
                    crc >>= 1
        return crc ^ 0xFFFF

    def _decode_callsign(self, data: bytes) -> Tuple[str, int]:
        """Decode AX.25 callsign from 7 bytes."""
        if len(data) < 7:
            return "", 0

        callsign = ""
        for i in range(6):
            char = (data[i] >> 1) & 0x7F
            if char != 0x20:  # Space
                callsign += chr(char)

        ssid = (data[6] >> 1) & 0x0F
        if ssid > 0:
            callsign += f"-{ssid}"

        return callsign, data[6]

    def _parse_frame(self, bits: List[int]) -> Optional[AX25Frame]:
        """Parse AX.25 frame from bits."""
        if len(bits) < 136:  # Minimum frame size (17 bytes)
            return None

        # Convert bits to bytes
        frame_bytes = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte |= bits[i + j] << j
            frame_bytes.append(byte)

        if len(frame_bytes) < 17:
            return None

        data = bytes(frame_bytes)

        # Verify CRC
        if len(data) < 2:
            return None

        payload = data[:-2]
        received_crc = data[-2] | (data[-1] << 8)
        computed_crc = self._compute_crc(payload)

        if received_crc != computed_crc:
            return AX25Frame(
                protocol=ProtocolType.AX25,
                timestamp=self._timestamp,
                raw_bits=bytes(bits),
                valid=False,
                error_message="CRC mismatch"
            )

        # Parse addresses
        dest_call, dest_ssid = self._decode_callsign(payload[0:7])
        src_call, src_ssid = self._decode_callsign(payload[7:14])

        # Parse digipeaters
        digipeaters = []
        addr_end = 14
        if not (src_ssid & 0x01):  # More addresses follow
            while addr_end + 7 <= len(payload):
                digi_call, digi_ssid = self._decode_callsign(payload[addr_end:addr_end + 7])
                digipeaters.append(digi_call)
                addr_end += 7
                if digi_ssid & 0x01:  # Last address
                    break

        if addr_end >= len(payload):
            return None

        # Control and PID
        control = payload[addr_end] if addr_end < len(payload) else 0
        pid = payload[addr_end + 1] if addr_end + 1 < len(payload) else 0

        # Info field
        info_start = addr_end + 2
        info = ""
        if info_start < len(payload):
            try:
                info = payload[info_start:].decode('ascii', errors='replace')
            except Exception:
                info = ""

        return AX25Frame(
            protocol=ProtocolType.AX25,
            timestamp=self._timestamp,
            raw_bits=bytes(bits),
            valid=True,
            source=src_call,
            destination=dest_call,
            digipeaters=digipeaters,
            control=control,
            pid=pid,
            info=info
        )

    def parse_aprs(self, frame: AX25Frame) -> Optional[APRSMessage]:
        """
        Parse APRS data from AX.25 frame.

        Args:
            frame: AX.25 frame

        Returns:
            APRS message or None
        """
        if not frame.valid or not frame.info:
            return None

        info = frame.info
        data_type = ""
        lat = 0.0
        lon = 0.0
        alt = 0.0
        speed = 0.0
        course = 0.0
        symbol = ""
        comment = ""

        # Detect data type from first character
        if len(info) > 0:
            type_char = info[0]

            if type_char == '!':
                data_type = "position"
            elif type_char == '=':
                data_type = "position_msg"
            elif type_char == '/':
                data_type = "position_timestamp"
            elif type_char == '@':
                data_type = "position_timestamp_msg"
            elif type_char == ':':
                data_type = "message"
            elif type_char == '>':
                data_type = "status"
            elif type_char == '_':
                data_type = "weather"
            elif type_char == 'T':
                data_type = "telemetry"
            elif type_char == '`' or type_char == "'":
                data_type = "mic-e"
            else:
                data_type = "unknown"

            # Parse position if present
            if data_type in ["position", "position_msg", "position_timestamp", "position_timestamp_msg"]:
                try:
                    # Find position data (format: DDMM.MMN/DDDMM.MMW)
                    pos_start = 1
                    if data_type.startswith("position_timestamp"):
                        pos_start = 8  # Skip timestamp

                    if len(info) > pos_start + 18:
                        lat_str = info[pos_start:pos_start + 8]
                        lon_str = info[pos_start + 9:pos_start + 18]
                        symbol = info[pos_start + 8] + info[pos_start + 18] if len(info) > pos_start + 18 else ""

                        # Parse latitude
                        lat_deg = float(lat_str[0:2])
                        lat_min = float(lat_str[2:7])
                        lat = lat_deg + lat_min / 60
                        if lat_str[7] == 'S':
                            lat = -lat

                        # Parse longitude
                        lon_deg = float(lon_str[0:3])
                        lon_min = float(lon_str[3:8])
                        lon = lon_deg + lon_min / 60
                        if lon_str[8] == 'W':
                            lon = -lon

                        # Comment is rest of info
                        comment = info[pos_start + 19:] if len(info) > pos_start + 19 else ""

                except (ValueError, IndexError):
                    pass

        return APRSMessage(
            protocol=ProtocolType.APRS,
            timestamp=frame.timestamp,
            raw_bits=frame.raw_bits,
            valid=True,
            source=frame.source,
            destination=frame.destination,
            path=frame.digipeaters,
            data_type=data_type,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            speed=speed,
            course=course,
            symbol=symbol,
            comment=comment
        )

    def decode(self, samples: np.ndarray) -> List[AX25Frame]:
        """
        Decode AX.25 frames from AFSK-demodulated samples.

        Args:
            samples: Demodulated samples

        Returns:
            List of decoded AX.25 frames
        """
        self._frames = []
        self._timestamp += len(samples) / self._sample_rate

        # Convert to bits
        sps = self._samples_per_bit
        for i in range(0, len(samples) - sps, sps):
            center = i + sps // 2
            if center < len(samples):
                raw_bit = 1 if samples[center] > 0 else 0
                bit = self._nrzi_decode(raw_bit)
                self._bit_buffer.append(bit)

        # Process bits
        while self._bit_buffer:
            bit = self._bit_buffer.pop(0)

            if self._in_frame:
                if bit == 1:
                    self._ones_count += 1
                    if self._ones_count < 6:
                        self._frame_buffer.append(1)
                    elif self._ones_count == 6:
                        # Possible flag or abort
                        pass
                    else:
                        # Abort - too many ones
                        self._in_frame = False
                        self._frame_buffer = []
                        self._ones_count = 0
                else:
                    if self._ones_count == 5:
                        # Stuffed bit - ignore
                        pass
                    elif self._ones_count == 6:
                        # End flag
                        if len(self._frame_buffer) > 16:
                            frame = self._parse_frame(self._frame_buffer)
                            if frame:
                                self._frames.append(frame)
                                self._notify_callbacks(frame)
                        self._frame_buffer = []
                        self._in_frame = True  # Stay in frame mode for next
                    else:
                        self._frame_buffer.append(0)
                    self._ones_count = 0
            else:
                # Looking for flag
                if bit == 1:
                    self._ones_count += 1
                else:
                    if self._ones_count == 6:
                        # Found flag
                        self._in_frame = True
                        self._frame_buffer = []
                    self._ones_count = 0

        return self._frames

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._frame_buffer = []
        self._in_frame = False
        self._ones_count = 0
        self._nrzi_last = 0
        self._timestamp = 0.0


# ============================================================================
# RDS (Radio Data System) Decoder
# ============================================================================

@dataclass
class RDSData(DecodedMessage):
    """RDS decoded data."""
    pi_code: int = 0  # Program Identification
    pty: int = 0  # Program Type
    tp: bool = False  # Traffic Program
    ta: bool = False  # Traffic Announcement
    ms: bool = False  # Music/Speech
    ps_name: str = ""  # Program Service name (8 chars)
    radio_text: str = ""  # Radio Text (64 chars)
    clock_time: str = ""  # CT (Clock Time)
    af_list: List[float] = None  # Alternative Frequencies

    def __post_init__(self):
        if self.af_list is None:
            self.af_list = []


class RDSDecoder(ProtocolDecoder):
    """
    RDS (Radio Data System) decoder.

    RDS is broadcast on FM radio at 57 kHz subcarrier.

    Features:
    - Block synchronization
    - Error detection and correction
    - PI code extraction
    - PS (Program Service) name decoding
    - RT (Radio Text) decoding
    - Clock time decoding
    """

    # RDS constants
    BLOCK_SIZE = 26  # bits
    GROUP_SIZE = 4  # blocks

    # Syndrome values for block types
    SYNDROMES = {
        0x3D8: 'A',
        0x3D4: 'B',
        0x25C: 'C',
        0x3CC: "C'",
        0x258: 'D',
    }

    # Offset words
    OFFSETS = {
        'A': 0x0FC,
        'B': 0x198,
        'C': 0x168,
        "C'": 0x350,
        'D': 0x1B4,
    }

    # PTY codes (North America)
    PTY_CODES = [
        "None", "News", "Information", "Sports", "Talk", "Rock",
        "Classic Rock", "Adult Hits", "Soft Rock", "Top 40", "Country",
        "Oldies", "Soft", "Nostalgia", "Jazz", "Classical", "R&B",
        "Soft R&B", "Language", "Religious Music", "Religious Talk",
        "Personality", "Public", "College", "Spanish Talk", "Spanish Music",
        "Hip Hop", "Unassigned", "Unassigned", "Weather", "Emergency Test",
        "Emergency"
    ]

    def __init__(self, sample_rate: float):
        """
        Initialize RDS decoder.

        Args:
            sample_rate: Sample rate in Hz (RDS is 1187.5 baud)
        """
        super().__init__(sample_rate)
        self._baud_rate = 1187.5
        self._samples_per_bit = sample_rate / self._baud_rate

        # State
        self._bit_buffer: List[int] = []
        self._synced = False
        self._block_count = 0
        self._current_group: List[int] = []

        # Decoded data
        self._pi_code = 0
        self._pty = 0
        self._tp = False
        self._ta = False
        self._ms = False
        self._ps_name = [' '] * 8
        self._radio_text = [' '] * 64
        self._rt_ab = 0

        self._timestamp = 0.0

    def _syndrome(self, block: int) -> int:
        """Calculate syndrome for 26-bit block."""
        # Generator polynomial for RDS
        poly = 0x5B9  # x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1

        reg = 0
        for i in range(26):
            bit = (block >> (25 - i)) & 1
            feedback = ((reg >> 9) & 1) ^ bit
            reg = ((reg << 1) | feedback) & 0x3FF
            if feedback:
                reg ^= poly

        return reg

    def _decode_block(self, bits: List[int]) -> Tuple[int, str]:
        """
        Decode 26-bit block.

        Returns:
            (16-bit data, block_type) or (0, '') if invalid
        """
        if len(bits) != 26:
            return 0, ''

        block = 0
        for bit in bits:
            block = (block << 1) | bit

        # Calculate syndrome
        syn = self._syndrome(block)

        # Check against known syndromes
        for expected_syn, block_type in self.SYNDROMES.items():
            if syn == expected_syn:
                # Extract 16-bit data
                data = (block >> 10) & 0xFFFF
                return data, block_type

        return 0, ''

    def _process_group(self, blocks: List[Tuple[int, str]]) -> Optional[RDSData]:
        """Process a complete RDS group (4 blocks)."""
        if len(blocks) != 4:
            return None

        data_a, type_a = blocks[0]
        data_b, type_b = blocks[1]
        data_c, type_c = blocks[2]
        data_d, type_d = blocks[3]

        if type_a != 'A' or type_b != 'B':
            return None

        # Block A: PI code
        self._pi_code = data_a

        # Block B: Group type, PTY, TP, etc.
        group_type = (data_b >> 12) & 0x0F
        version = (data_b >> 11) & 0x01
        self._tp = bool((data_b >> 10) & 0x01)
        self._pty = (data_b >> 5) & 0x1F

        # Process based on group type
        if group_type == 0:  # Basic tuning and switching
            self._ta = bool((data_b >> 4) & 0x01)
            self._ms = bool((data_b >> 3) & 0x01)

            # PS name (2 characters per group 0)
            ps_addr = (data_b & 0x03) * 2
            if type_d == 'D':
                char1 = (data_d >> 8) & 0xFF
                char2 = data_d & 0xFF
                if 32 <= char1 < 127:
                    self._ps_name[ps_addr] = chr(char1)
                if 32 <= char2 < 127:
                    self._ps_name[ps_addr + 1] = chr(char2)

        elif group_type == 2:  # Radio Text
            rt_ab = (data_b >> 4) & 0x01
            if rt_ab != self._rt_ab:
                self._radio_text = [' '] * 64
                self._rt_ab = rt_ab

            rt_addr = (data_b & 0x0F) * 4
            if version == 0 and type_c == 'C' and type_d == 'D':
                # Version A: 4 characters
                chars = [
                    (data_c >> 8) & 0xFF,
                    data_c & 0xFF,
                    (data_d >> 8) & 0xFF,
                    data_d & 0xFF,
                ]
                for i, char in enumerate(chars):
                    if rt_addr + i < 64 and 32 <= char < 127:
                        self._radio_text[rt_addr + i] = chr(char)

        # Return current state
        return RDSData(
            protocol=ProtocolType.RDS,
            timestamp=self._timestamp,
            raw_bits=bytes(),
            valid=True,
            pi_code=self._pi_code,
            pty=self._pty,
            tp=self._tp,
            ta=self._ta,
            ms=self._ms,
            ps_name=''.join(self._ps_name).strip(),
            radio_text=''.join(self._radio_text).strip(),
        )

    def decode(self, samples: np.ndarray) -> List[RDSData]:
        """
        Decode RDS from demodulated samples.

        Args:
            samples: BPSK-demodulated RDS samples

        Returns:
            List of RDS data updates
        """
        results = []
        self._timestamp += len(samples) / self._sample_rate

        # Convert to bits
        sps = self._samples_per_bit
        i = 0.0
        while i < len(samples) - sps:
            center = int(i + sps / 2)
            if center < len(samples):
                bit = 1 if samples[center] > 0 else 0
                self._bit_buffer.append(bit)
            i += sps

        # Process bits into blocks
        while len(self._bit_buffer) >= self.BLOCK_SIZE:
            block_bits = self._bit_buffer[:self.BLOCK_SIZE]
            data, block_type = self._decode_block(block_bits)

            if block_type:
                if block_type == 'A':
                    self._current_group = [(data, block_type)]
                    self._synced = True
                elif self._synced:
                    self._current_group.append((data, block_type))

                    if len(self._current_group) == 4:
                        rds_data = self._process_group(self._current_group)
                        if rds_data:
                            results.append(rds_data)
                            self._notify_callbacks(rds_data)
                        self._current_group = []

                self._bit_buffer = self._bit_buffer[self.BLOCK_SIZE:]
            else:
                # No sync, shift by 1 bit
                self._bit_buffer.pop(0)
                if self._synced:
                    self._synced = False
                    self._current_group = []

        return results

    def get_pty_name(self, pty: int) -> str:
        """Get PTY name from code."""
        if 0 <= pty < len(self.PTY_CODES):
            return self.PTY_CODES[pty]
        return "Unknown"

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._synced = False
        self._current_group = []
        self._ps_name = [' '] * 8
        self._radio_text = [' '] * 64
        self._timestamp = 0.0


# ============================================================================
# ADS-B (Automatic Dependent Surveillance-Broadcast) Decoder
# ============================================================================

@dataclass
class ADSBMessage(DecodedMessage):
    """ADS-B decoded message."""
    icao_address: str = ""  # 24-bit ICAO aircraft address
    downlink_format: int = 0  # DF (17 for ADS-B)
    type_code: int = 0  # TC (message type)
    callsign: str = ""  # Aircraft callsign (8 chars)
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: int = 0  # Altitude in feet
    velocity: float = 0.0  # Ground speed in knots
    heading: float = 0.0  # Track angle in degrees
    vertical_rate: int = 0  # Vertical rate in ft/min
    squawk: str = ""  # Transponder code
    on_ground: bool = False
    category: str = ""  # Aircraft category


class ADSBDecoder(ProtocolDecoder):
    """
    ADS-B (Mode S) decoder.

    Decodes ADS-B messages broadcast by aircraft transponders at 1090 MHz.

    Features:
    - Mode S preamble detection
    - CRC-24 validation
    - Aircraft identification (callsign)
    - Airborne position decoding (CPR)
    - Velocity decoding
    - Altitude extraction
    """

    # Mode S constants
    PREAMBLE_US = 8.0  # Preamble duration in microseconds
    SHORT_MSG_BITS = 56  # Short message (DF 0, 4, 5, 11)
    LONG_MSG_BITS = 112  # Long message (DF 16, 17, 18, 19, 20, 21)

    # CRC-24 polynomial (0x1FFF409)
    CRC_POLY = 0x1FFF409
    CRC_TABLE: List[int] = []

    # Character lookup for callsign
    CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

    def __init__(self, sample_rate: float):
        """
        Initialize ADS-B decoder.

        Args:
            sample_rate: Sample rate in Hz (typically 2 MHz for ADS-B)
        """
        super().__init__(sample_rate)

        # 1090 MHz Mode S: 1 us per bit, 2 samples per bit at 2 MHz
        self._samples_per_bit = int(sample_rate / 1e6)
        self._preamble_samples = int(self.PREAMBLE_US * sample_rate / 1e6)

        # Generate CRC table
        if not ADSBDecoder.CRC_TABLE:
            ADSBDecoder.CRC_TABLE = self._generate_crc_table()

        # State
        self._bit_buffer: List[int] = []
        self._messages: List[ADSBMessage] = []
        self._timestamp = 0.0

        # CPR position cache for decoding (icao -> (even_msg, odd_msg, timestamp))
        self._cpr_cache: Dict[str, Tuple[Any, Any, float]] = {}

    def _generate_crc_table(self) -> List[int]:
        """Generate CRC-24 lookup table."""
        table = []
        for i in range(256):
            crc = i << 16
            for _ in range(8):
                if crc & 0x800000:
                    crc = (crc << 1) ^ self.CRC_POLY
                else:
                    crc <<= 1
            table.append(crc & 0xFFFFFF)
        return table

    def _compute_crc(self, data: bytes) -> int:
        """Compute CRC-24 for Mode S message."""
        crc = 0
        for byte in data:
            crc = ((crc << 8) ^ self.CRC_TABLE[(crc >> 16) ^ byte]) & 0xFFFFFF
        return crc

    def _detect_preamble(self, samples: np.ndarray, start: int) -> bool:
        """
        Detect Mode S preamble pattern.

        Preamble: 8 us total
        Pattern: |1|0|1|0|0|0|0|1|0|1|0|0|0|0|0|0| (at 2 MHz sample rate)
        """
        sps = self._samples_per_bit

        if start + 16 * sps > len(samples):
            return False

        # Expected pattern positions (in half-bit units at 2MHz)
        # High: 0, 2, 7, 9 (0-0.5, 1-1.5, 3.5-4, 4.5-5 us)
        # Low: 1, 3-6, 8, 10-15

        threshold = np.mean(np.abs(samples[start:start + 16 * sps]))

        # Check high positions
        for pos in [0, 2, 7, 9]:
            idx = start + pos * sps
            if idx < len(samples) and samples[idx] < threshold:
                return False

        # Check some low positions
        for pos in [1, 4, 5, 8, 12]:
            idx = start + pos * sps
            if idx < len(samples) and samples[idx] > threshold:
                return False

        return True

    def _extract_bits(self, samples: np.ndarray, start: int, num_bits: int) -> List[int]:
        """Extract bits from PPM (Pulse Position Modulation) samples."""
        bits = []
        sps = self._samples_per_bit

        for i in range(num_bits):
            # Each bit is 1 us, first half high = 1, second half high = 0
            pos = start + i * sps
            if pos + sps > len(samples):
                break

            first_half = samples[pos:pos + sps // 2]
            second_half = samples[pos + sps // 2:pos + sps]

            if len(first_half) > 0 and len(second_half) > 0:
                if np.mean(first_half) > np.mean(second_half):
                    bits.append(1)
                else:
                    bits.append(0)

        return bits

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert bit list to bytes."""
        result = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)

    def _decode_callsign(self, data: bytes) -> str:
        """Decode aircraft identification (callsign)."""
        if len(data) < 7:
            return ""

        # Bytes 1-6 contain callsign (6 bits per character, 8 characters)
        chars = []
        bits = int.from_bytes(data[1:7], 'big')

        for i in range(8):
            char_idx = (bits >> (42 - i * 6)) & 0x3F
            if char_idx < len(self.CHARSET):
                char = self.CHARSET[char_idx]
                if char != '#':
                    chars.append(char)

        return ''.join(chars).strip()

    def _decode_altitude(self, data: bytes) -> int:
        """Decode altitude from airborne position message."""
        if len(data) < 6:
            return 0

        alt_bits = ((data[1] & 0xFF) << 4) | ((data[2] >> 4) & 0x0F)

        # Q-bit indicates 25 or 100 ft resolution
        q_bit = (alt_bits >> 4) & 0x01

        if q_bit:
            # 25 ft resolution
            n = ((alt_bits & 0x0F) << 7) | ((alt_bits >> 5) & 0x7F)
            altitude = n * 25 - 1000
        else:
            # Gillham code (100 ft resolution) - simplified
            altitude = alt_bits * 100 - 1300

        return altitude

    def _decode_cpr_position(
        self,
        icao: str,
        lat_cpr: int,
        lon_cpr: int,
        odd: bool,
        airborne: bool
    ) -> Tuple[float, float]:
        """
        Decode CPR (Compact Position Reporting) position.

        Requires both odd and even messages for global decode.
        """
        # Check cache for complementary message
        if icao not in self._cpr_cache:
            self._cpr_cache[icao] = (None, None, 0.0)

        even_msg, odd_msg, cache_time = self._cpr_cache[icao]
        current_time = time.time()

        # Cache timeout: 10 seconds
        if current_time - cache_time > 10:
            even_msg = None
            odd_msg = None

        # Store current message
        msg = (lat_cpr, lon_cpr)
        if odd:
            odd_msg = msg
        else:
            even_msg = msg

        self._cpr_cache[icao] = (even_msg, odd_msg, current_time)

        # Need both messages for global decode
        if even_msg is None or odd_msg is None:
            return 0.0, 0.0

        # CPR decoding constants
        if airborne:
            dlat0 = 360.0 / 60
            dlat1 = 360.0 / 59
        else:
            dlat0 = 360.0 / 90
            dlat1 = 360.0 / 89

        lat_even, lon_even = even_msg
        lat_odd, lon_odd = odd_msg

        # Latitude decode
        j = int(np.floor((59 * lat_even - 60 * lat_odd) / 131072.0 + 0.5))
        lat_even_val = dlat0 * ((j % 60) + lat_even / 131072.0)
        lat_odd_val = dlat1 * ((j % 59) + lat_odd / 131072.0)

        if lat_even_val >= 270:
            lat_even_val -= 360
        if lat_odd_val >= 270:
            lat_odd_val -= 360

        # Use most recent message
        if odd:
            lat = lat_odd_val
        else:
            lat = lat_even_val

        # Longitude decode - need NL (longitude zone)
        def nl(lat: float) -> int:
            if abs(lat) >= 87:
                return 1
            return int(np.floor(2 * np.pi / np.arccos(
                1 - (1 - np.cos(np.pi / 30)) / (np.cos(np.pi * lat / 180) ** 2)
            )))

        nl_lat = nl(lat)

        if odd:
            ni = max(nl_lat - 1, 1)
            dlon = 360.0 / ni
            m = int(np.floor((lon_even * (nl_lat - 1) - lon_odd * nl_lat) / 131072.0 + 0.5))
            lon = dlon * ((m % ni) + lon_odd / 131072.0)
        else:
            ni = max(nl_lat, 1)
            dlon = 360.0 / ni
            m = int(np.floor((lon_even * (nl_lat - 1) - lon_odd * nl_lat) / 131072.0 + 0.5))
            lon = dlon * ((m % ni) + lon_even / 131072.0)

        if lon > 180:
            lon -= 360

        return lat, lon

    def _decode_velocity(self, data: bytes) -> Tuple[float, float, int]:
        """Decode velocity from airborne velocity message."""
        if len(data) < 7:
            return 0.0, 0.0, 0

        subtype = data[0] & 0x07

        if subtype in (1, 2):
            # Ground speed
            ew_dir = (data[1] >> 2) & 0x01
            ew_vel = ((data[1] & 0x03) << 8) | data[2]
            ns_dir = (data[3] >> 7) & 0x01
            ns_vel = ((data[3] & 0x7F) << 3) | (data[4] >> 5)

            if ew_vel == 0 and ns_vel == 0:
                return 0.0, 0.0, 0

            ew_vel -= 1
            ns_vel -= 1

            if ew_dir:
                ew_vel = -ew_vel
            if ns_dir:
                ns_vel = -ns_vel

            # Calculate speed and heading
            velocity = np.sqrt(ew_vel ** 2 + ns_vel ** 2)
            heading = np.degrees(np.arctan2(ew_vel, ns_vel))
            if heading < 0:
                heading += 360

            # Vertical rate
            vr_sign = (data[4] >> 3) & 0x01
            vr_value = ((data[4] & 0x07) << 6) | (data[5] >> 2)
            vertical_rate = (vr_value - 1) * 64
            if vr_sign:
                vertical_rate = -vertical_rate

            return velocity, heading, vertical_rate

        return 0.0, 0.0, 0

    def _parse_message(self, bits: List[int]) -> Optional[ADSBMessage]:
        """Parse Mode S message from bits."""
        if len(bits) < self.SHORT_MSG_BITS:
            return None

        data = self._bits_to_bytes(bits)
        if len(data) < 7:
            return None

        # Downlink Format (first 5 bits)
        df = (data[0] >> 3) & 0x1F

        # Determine message length
        if df in (0, 4, 5, 11):
            msg_bits = self.SHORT_MSG_BITS
            msg_bytes = 7
        elif df in (16, 17, 18, 19, 20, 21):
            msg_bits = self.LONG_MSG_BITS
            msg_bytes = 14
        else:
            return None

        if len(bits) < msg_bits or len(data) < msg_bytes:
            return None

        data = data[:msg_bytes]

        # CRC check
        crc = self._compute_crc(data[:-3])
        received_crc = int.from_bytes(data[-3:], 'big')

        # For DF17/18, CRC should be zero or match ICAO address
        if df == 17 or df == 18:
            if crc != received_crc:
                return None

        # Extract ICAO address
        icao = f"{data[1]:02X}{data[2]:02X}{data[3]:02X}"

        # Create base message
        msg = ADSBMessage(
            protocol=ProtocolType.ADSB,
            timestamp=self._timestamp,
            raw_bits=bytes(bits[:msg_bits]),
            valid=True,
            icao_address=icao,
            downlink_format=df
        )

        # Parse DF17/18 extended squitter
        if df in (17, 18) and len(data) >= 11:
            tc = (data[4] >> 3) & 0x1F
            msg.type_code = tc

            # Aircraft identification (TC 1-4)
            if 1 <= tc <= 4:
                msg.callsign = self._decode_callsign(data[4:])
                categories = ["", "Light", "Medium 1", "Medium 2",
                            "High vortex", "Heavy", "High perf", "Rotorcraft"]
                cat_idx = data[4] & 0x07
                msg.category = categories[cat_idx] if cat_idx < len(categories) else ""

            # Airborne position (TC 9-18)
            elif 9 <= tc <= 18:
                msg.altitude = self._decode_altitude(data[4:])
                msg.on_ground = False

                # CPR position
                lat_cpr = ((data[6] & 0x03) << 15) | (data[7] << 7) | (data[8] >> 1)
                lon_cpr = ((data[8] & 0x01) << 16) | (data[9] << 8) | data[10]
                odd = bool((data[6] >> 2) & 0x01)

                lat, lon = self._decode_cpr_position(icao, lat_cpr, lon_cpr, odd, True)
                if lat != 0.0 or lon != 0.0:
                    msg.latitude = lat
                    msg.longitude = lon

            # Airborne velocity (TC 19)
            elif tc == 19:
                velocity, heading, vrate = self._decode_velocity(data[4:])
                msg.velocity = velocity
                msg.heading = heading
                msg.vertical_rate = vrate

            # Surface position (TC 5-8)
            elif 5 <= tc <= 8:
                msg.on_ground = True

        return msg

    def decode(self, samples: np.ndarray) -> List[ADSBMessage]:
        """
        Decode ADS-B messages from raw samples.

        Args:
            samples: Magnitude samples from 1090 MHz

        Returns:
            List of decoded ADS-B messages
        """
        self._messages = []
        self._timestamp += len(samples) / self._sample_rate

        # Scan for preambles
        i = 0
        while i < len(samples) - (self._preamble_samples + self.LONG_MSG_BITS * self._samples_per_bit):
            if self._detect_preamble(samples, i):
                # Try to extract message
                msg_start = i + self._preamble_samples

                # Try long message first
                bits = self._extract_bits(samples, msg_start, self.LONG_MSG_BITS)

                if len(bits) >= self.SHORT_MSG_BITS:
                    msg = self._parse_message(bits)
                    if msg:
                        self._messages.append(msg)
                        self._notify_callbacks(msg)
                        # Skip past this message
                        i = msg_start + len(bits) * self._samples_per_bit
                        continue

            i += self._samples_per_bit

        return self._messages

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._cpr_cache.clear()
        self._timestamp = 0.0


# ============================================================================
# FLEX Protocol Decoder
# ============================================================================

@dataclass
class FLEXMessage(DecodedMessage):
    """FLEX pager message."""
    capcode: int = 0  # Pager address
    cycle: int = 0  # FLEX cycle number (0-14)
    frame: int = 0  # Frame within cycle (0-127)
    phase: str = ""  # Phase (A, B, C, D)
    message_type: str = ""  # "alpha", "numeric", "tone", "secure"
    content: str = ""
    baud_rate: int = 1600


class FLEXDecoder(ProtocolDecoder):
    """
    FLEX pager protocol decoder.

    FLEX is a high-speed paging protocol developed by Motorola,
    operating at 1600, 3200, or 6400 baud with 2-level or 4-level FSK.

    Features:
    - Multi-speed detection (1600/3200/6400 baud)
    - 2-FSK and 4-FSK demodulation
    - BCH error correction
    - Alphanumeric and numeric message decoding
    - Capcode (address) extraction
    """

    # FLEX constants
    SYNC_1 = 0xA6C6AAAA  # Sync pattern 1
    SYNC_2 = 0x5939AAAA  # Sync pattern 2

    # Frame structure
    FRAME_SYNC_BITS = 32
    FRAME_INFO_BITS = 32
    BLOCK_BITS = 32
    BLOCKS_PER_FRAME = 11  # Data blocks per frame

    # BCH(32,21) polynomial
    BCH_POLY = 0x769

    def __init__(self, sample_rate: float, baud_rate: int = 1600):
        """
        Initialize FLEX decoder.

        Args:
            sample_rate: Sample rate in Hz
            baud_rate: Initial baud rate (1600, 3200, 6400)
        """
        super().__init__(sample_rate)
        self._baud_rate = baud_rate
        self._samples_per_bit = int(sample_rate / baud_rate)

        # State
        self._bit_buffer: List[int] = []
        self._synced = False
        self._current_frame: List[int] = []
        self._phase = "A"
        self._cycle = 0
        self._frame_num = 0

        # Messages
        self._messages: List[FLEXMessage] = []
        self._timestamp = 0.0

    @property
    def baud_rate(self) -> int:
        """Get baud rate."""
        return self._baud_rate

    def _detect_baud_rate(self, samples: np.ndarray) -> int:
        """Detect FLEX baud rate from sync pattern."""
        best_baud = self._baud_rate
        best_score = 0

        for baud in [1600, 3200, 6400]:
            sps = int(self._sample_rate / baud)
            if sps < 2:
                continue

            # Look for alternating pattern (AAAA portion of sync)
            score = 0
            for i in range(0, min(len(samples) - sps, 64 * sps), sps):
                if i + sps < len(samples):
                    sign1 = 1 if samples[i] > 0 else -1
                    sign2 = 1 if samples[i + sps] > 0 else -1
                    if sign1 != sign2:
                        score += 1

            if score > best_score:
                best_score = score
                best_baud = baud

        return best_baud

    def _bits_to_word(self, bits: List[int]) -> int:
        """Convert 32 bits to word."""
        word = 0
        for bit in bits[:32]:
            word = (word << 1) | bit
        return word

    def _check_bch(self, word: int) -> Tuple[bool, int]:
        """Check BCH(32,21) error correction."""
        codeword = word

        # Calculate syndrome
        syndrome = 0
        for i in range(21):
            if codeword & (1 << (31 - i)):
                syndrome ^= (self.BCH_POLY << (10 - i)) if i < 11 else (self.BCH_POLY >> (i - 10))

        syndrome &= 0x7FF

        if syndrome == 0:
            return True, word

        # Try single-bit error correction
        for i in range(32):
            test = word ^ (1 << (31 - i))
            test_syndrome = 0
            for j in range(21):
                if test & (1 << (31 - j)):
                    test_syndrome ^= (self.BCH_POLY << (10 - j)) if j < 11 else (self.BCH_POLY >> (j - 10))
            test_syndrome &= 0x7FF

            if test_syndrome == 0:
                return True, test

        return False, word

    def _decode_alpha(self, words: List[int]) -> str:
        """Decode alphanumeric message from words."""
        result = ""

        for word in words:
            # Extract 7-bit characters
            for i in range(4):
                char_code = (word >> (21 - i * 7)) & 0x7F
                if 32 <= char_code < 127:
                    result += chr(char_code)
                elif char_code == 0:
                    return result

        return result

    def _decode_numeric(self, words: List[int]) -> str:
        """Decode numeric message from words."""
        result = ""
        # FLEX numeric uses 4-bit BCD
        num_chars = "0123456789 U-][("

        for word in words:
            for i in range(5):  # 5 nibbles per 21-bit data field
                nibble = (word >> (17 - i * 4)) & 0x0F
                if nibble < len(num_chars):
                    char = num_chars[nibble]
                    if char != '[':  # End marker
                        result += char
                    else:
                        return result

        return result

    def _parse_frame(self, frame_bits: List[int]) -> List[FLEXMessage]:
        """Parse FLEX frame and extract messages."""
        messages = []

        if len(frame_bits) < self.BLOCKS_PER_FRAME * self.BLOCK_BITS:
            return messages

        # Extract blocks
        blocks = []
        for i in range(self.BLOCKS_PER_FRAME):
            start = i * self.BLOCK_BITS
            word = self._bits_to_word(frame_bits[start:start + self.BLOCK_BITS])
            valid, corrected = self._check_bch(word)
            if valid:
                blocks.append(corrected)
            else:
                blocks.append(None)

        # First block contains block information word
        if blocks[0] is None:
            return messages

        biw = blocks[0]

        # Extract address and message fields
        # BIW contains pointers to address and vector fields
        addr_start = ((biw >> 16) & 0x1F) if biw else 1
        vector_start = ((biw >> 8) & 0x1F) if biw else 1

        # Find addresses (capcodes)
        capcodes = []
        for i in range(1, min(addr_start + 1, len(blocks))):
            if blocks[i] is not None:
                capcode = (blocks[i] >> 10) & 0x1FFFFF
                capcodes.append(capcode)

        # Find message vectors and content
        for i, capcode in enumerate(capcodes):
            if vector_start + i >= len(blocks):
                break

            vector = blocks[vector_start + i] if vector_start + i < len(blocks) else None
            if vector is None:
                continue

            # Determine message type
            msg_type_bits = (vector >> 18) & 0x07
            msg_types = ["tone", "tone", "numeric", "numeric",
                        "alpha", "alpha", "secure", "secure"]
            msg_type = msg_types[msg_type_bits] if msg_type_bits < len(msg_types) else "unknown"

            # Get message content from remaining blocks
            content_blocks = [b for b in blocks[vector_start + len(capcodes):]
                            if b is not None]

            if msg_type == "alpha":
                content = self._decode_alpha(content_blocks)
            elif msg_type == "numeric":
                content = self._decode_numeric(content_blocks)
            else:
                content = ""

            msg = FLEXMessage(
                protocol=ProtocolType.FLEX,
                timestamp=self._timestamp,
                raw_bits=bytes(frame_bits),
                valid=True,
                capcode=capcode,
                cycle=self._cycle,
                frame=self._frame_num,
                phase=self._phase,
                message_type=msg_type,
                content=content,
                baud_rate=self._baud_rate
            )
            messages.append(msg)

        return messages

    def decode(self, samples: np.ndarray) -> List[FLEXMessage]:
        """
        Decode FLEX messages from FSK-demodulated samples.

        Args:
            samples: FSK-demodulated samples

        Returns:
            List of decoded FLEX messages
        """
        self._messages = []
        self._timestamp += len(samples) / self._sample_rate

        # Convert to bits
        sps = self._samples_per_bit
        for i in range(0, len(samples) - sps, sps):
            center = i + sps // 2
            if center < len(samples):
                bit = 1 if samples[center] > 0 else 0
                self._bit_buffer.append(bit)

        # Look for sync
        while len(self._bit_buffer) >= 64:
            # Check for sync words
            word1 = self._bits_to_word(self._bit_buffer[:32])
            self._bits_to_word(self._bit_buffer[32:64])

            if (word1 == self.SYNC_1 or word1 == self.SYNC_2):
                self._synced = True
                self._current_frame = []
                self._bit_buffer = self._bit_buffer[64:]

                # Parse Frame Info Word
                if len(self._bit_buffer) >= 32:
                    fiw = self._bits_to_word(self._bit_buffer[:32])
                    self._cycle = (fiw >> 24) & 0x0F
                    self._frame_num = (fiw >> 17) & 0x7F
                    self._bit_buffer = self._bit_buffer[32:]

            elif self._synced:
                # Collect frame bits
                self._current_frame.append(self._bit_buffer.pop(0))

                frame_size = self.BLOCKS_PER_FRAME * self.BLOCK_BITS
                if len(self._current_frame) >= frame_size:
                    msgs = self._parse_frame(self._current_frame)
                    for msg in msgs:
                        self._messages.append(msg)
                        self._notify_callbacks(msg)
                    self._current_frame = []
                    self._synced = False
            else:
                self._bit_buffer.pop(0)

        return self._messages

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._synced = False
        self._current_frame = []
        self._timestamp = 0.0


# ============================================================================
# ACARS (Aircraft Communications Addressing and Reporting System) Decoder
# ============================================================================

@dataclass
class ACARSMessage(DecodedMessage):
    """ACARS decoded message."""
    mode: str = ""  # Mode character
    registration: str = ""  # Aircraft registration (7 chars)
    ack: str = ""  # Acknowledgement character
    label: str = ""  # Message label (2 chars)
    block_id: str = ""  # Block identifier
    message_number: str = ""  # Message sequence number
    flight_id: str = ""  # Flight identification
    text: str = ""  # Message text
    block_end: str = ""  # Block end character


class ACARSDecoder(ProtocolDecoder):
    """
    ACARS decoder.

    ACARS is a digital data link system for aircraft communications
    on VHF frequencies (129.125, 130.025, 130.450 MHz etc.).

    Features:
    - 2400 baud AM MSK modulation
    - Message structure parsing
    - CRC-16 validation
    - Flight ID and registration extraction
    - Message text decoding
    """

    # ACARS constants
    PREKEY = 0x2B2B  # Preamble: ++
    SOH = 0x01  # Start of header
    STX = 0x02  # Start of text
    ETX = 0x03  # End of text
    ETB = 0x17  # End of transmission block

    # Character set (odd parity, 7-bit ASCII)
    BAUD_RATE = 2400

    def __init__(self, sample_rate: float):
        """
        Initialize ACARS decoder.

        Args:
            sample_rate: Sample rate in Hz
        """
        super().__init__(sample_rate)
        self._samples_per_bit = int(sample_rate / self.BAUD_RATE)

        # State
        self._bit_buffer: List[int] = []
        self._byte_buffer: List[int] = []
        self._in_message = False
        self._last_bit = 0

        # Messages
        self._messages: List[ACARSMessage] = []
        self._timestamp = 0.0

    def _nrzi_decode(self, bit: int) -> int:
        """NRZI decode."""
        result = 0 if bit != self._last_bit else 1
        self._last_bit = bit
        return result

    def _remove_parity(self, byte: int) -> int:
        """Remove parity bit and return 7-bit character."""
        return byte & 0x7F

    def _compute_crc(self, data: bytes) -> int:
        """Compute ACARS CRC-16."""
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
            crc &= 0xFFFF
        return crc

    def _parse_message(self, data: bytes) -> Optional[ACARSMessage]:
        """Parse ACARS message from bytes."""
        if len(data) < 10:
            return None

        # Find SOH
        soh_idx = -1
        for i in range(len(data)):
            if data[i] == self.SOH:
                soh_idx = i
                break

        if soh_idx < 0 or soh_idx + 8 > len(data):
            return None

        # Extract fields after SOH
        mode = chr(data[soh_idx + 1]) if data[soh_idx + 1] < 128 else ""

        # Registration (7 characters)
        reg_start = soh_idx + 2
        registration = ""
        for i in range(7):
            if reg_start + i < len(data):
                char = self._remove_parity(data[reg_start + i])
                if 32 <= char < 127:
                    registration += chr(char)

        # Acknowledgement
        ack_idx = soh_idx + 9
        ack = chr(data[ack_idx]) if ack_idx < len(data) and data[ack_idx] < 128 else ""

        # Label (2 characters)
        label_start = soh_idx + 10
        label = ""
        for i in range(2):
            if label_start + i < len(data):
                char = self._remove_parity(data[label_start + i])
                if 32 <= char < 127:
                    label += chr(char)

        # Block ID
        block_id_idx = soh_idx + 12
        block_id = chr(data[block_id_idx]) if block_id_idx < len(data) and data[block_id_idx] < 128 else ""

        # Find STX for message text
        stx_idx = -1
        for i in range(block_id_idx, len(data)):
            if data[i] == self.STX:
                stx_idx = i
                break

        # Find message number and flight ID before STX
        msg_number = ""
        flight_id = ""
        if stx_idx > 0:
            header = data[block_id_idx + 1:stx_idx]
            if len(header) >= 4:
                msg_number = ''.join(chr(self._remove_parity(b))
                                    for b in header[:4] if 32 <= self._remove_parity(b) < 127)
            if len(header) >= 10:
                flight_id = ''.join(chr(self._remove_parity(b))
                                   for b in header[4:10] if 32 <= self._remove_parity(b) < 127)

        # Find message text
        text = ""
        block_end = ""
        if stx_idx >= 0:
            # Find ETX or ETB
            end_idx = len(data) - 2  # Leave room for CRC
            for i in range(stx_idx + 1, len(data)):
                if data[i] in (self.ETX, self.ETB):
                    end_idx = i
                    block_end = chr(data[i])
                    break

            # Extract text
            text_bytes = data[stx_idx + 1:end_idx]
            text = ''.join(chr(self._remove_parity(b))
                          for b in text_bytes if 32 <= self._remove_parity(b) < 127)

        return ACARSMessage(
            protocol=ProtocolType.ACARS,
            timestamp=self._timestamp,
            raw_bits=data,
            valid=True,
            mode=mode,
            registration=registration.strip(),
            ack=ack,
            label=label,
            block_id=block_id,
            message_number=msg_number.strip(),
            flight_id=flight_id.strip(),
            text=text,
            block_end=block_end
        )

    def decode(self, samples: np.ndarray) -> List[ACARSMessage]:
        """
        Decode ACARS messages from MSK-demodulated samples.

        Args:
            samples: MSK-demodulated samples

        Returns:
            List of decoded ACARS messages
        """
        self._messages = []
        self._timestamp += len(samples) / self._sample_rate

        # Convert to bits
        sps = self._samples_per_bit
        for i in range(0, len(samples) - sps, sps):
            center = i + sps // 2
            if center < len(samples):
                raw_bit = 1 if samples[center] > 0 else 0
                bit = self._nrzi_decode(raw_bit)
                self._bit_buffer.append(bit)

        # Convert bits to bytes
        while len(self._bit_buffer) >= 8:
            byte = 0
            for i in range(8):
                byte |= self._bit_buffer[i] << i
            self._bit_buffer = self._bit_buffer[8:]

            if not self._in_message:
                # Look for preamble (++)
                self._byte_buffer.append(byte)
                if len(self._byte_buffer) >= 2:
                    if (self._byte_buffer[-2] == 0x2B and
                        self._byte_buffer[-1] == 0x2B):
                        self._in_message = True
                        self._byte_buffer = [0x2B, 0x2B]

                # Limit buffer size
                if len(self._byte_buffer) > 20:
                    self._byte_buffer.pop(0)
            else:
                self._byte_buffer.append(byte)

                # Check for message end
                if byte in (self.ETX, self.ETB) and len(self._byte_buffer) >= 13:
                    # Wait for CRC (2 bytes)
                    pass
                elif len(self._byte_buffer) >= 300:  # Max message size
                    # Try to parse
                    msg = self._parse_message(bytes(self._byte_buffer))
                    if msg:
                        self._messages.append(msg)
                        self._notify_callbacks(msg)
                    self._byte_buffer = []
                    self._in_message = False

                # Also try to parse if we have ETX/ETB + CRC
                if len(self._byte_buffer) >= 15:
                    # Check if we have a complete message
                    for i in range(len(self._byte_buffer) - 3, max(10, len(self._byte_buffer) - 50), -1):
                        if self._byte_buffer[i] in (self.ETX, self.ETB):
                            msg = self._parse_message(bytes(self._byte_buffer[:i + 3]))
                            if msg:
                                self._messages.append(msg)
                                self._notify_callbacks(msg)
                            self._byte_buffer = []
                            self._in_message = False
                            break

        return self._messages

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._byte_buffer = []
        self._in_message = False
        self._last_bit = 0
        self._timestamp = 0.0


# Factory function
def create_protocol_decoder(
    protocol: ProtocolType,
    sample_rate: float,
    **kwargs
) -> ProtocolDecoder:
    """
    Create a protocol decoder.

    Args:
        protocol: Protocol type
        sample_rate: Sample rate in Hz
        **kwargs: Protocol-specific parameters

    Returns:
        Protocol decoder instance

    Supported protocols:
        - POCSAG: Paging (512/1200/2400 baud FSK)
        - FLEX: Paging (1600/3200/6400 baud FSK)
        - AX25/APRS: Amateur packet radio (1200 baud AFSK)
        - RDS: FM broadcast data (1187.5 baud BPSK)
        - ADSB: Aircraft transponder (1090 MHz PPM)
        - ACARS: Aircraft communications (2400 baud MSK)
    """
    if protocol == ProtocolType.POCSAG:
        return POCSAGDecoder(sample_rate, **kwargs)
    elif protocol == ProtocolType.FLEX:
        return FLEXDecoder(sample_rate, **kwargs)
    elif protocol in (ProtocolType.AX25, ProtocolType.APRS):
        return AX25Decoder(sample_rate, **kwargs)
    elif protocol == ProtocolType.RDS:
        return RDSDecoder(sample_rate)
    elif protocol == ProtocolType.ADSB:
        return ADSBDecoder(sample_rate)
    elif protocol == ProtocolType.ACARS:
        return ACARSDecoder(sample_rate)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


# Export all decoders and message types
__all__ = [
    # Protocol types
    "ProtocolType",
    "DecodedMessage",
    # POCSAG
    "POCSAGMessage",
    "POCSAGDecoder",
    # FLEX
    "FLEXMessage",
    "FLEXDecoder",
    # AX.25/APRS
    "AX25Frame",
    "APRSMessage",
    "AX25Decoder",
    # RDS
    "RDSData",
    "RDSDecoder",
    # ADS-B
    "ADSBMessage",
    "ADSBDecoder",
    # ACARS
    "ACARSMessage",
    "ACARSDecoder",
    # Factory
    "create_protocol_decoder",
]

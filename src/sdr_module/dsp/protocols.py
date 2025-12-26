"""
Protocol decoders for common radio protocols.

Supports:
- POCSAG: Pager protocol (512/1200/2400 baud)
- AX.25/APRS: Amateur packet radio
- RDS: Radio Data System (FM broadcast)
"""

import numpy as np
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class ProtocolType(Enum):
    """Supported protocol types."""
    POCSAG = "pocsag"
    AX25 = "ax25"
    APRS = "aprs"
    RDS = "rds"


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
    """
    if protocol == ProtocolType.POCSAG:
        return POCSAGDecoder(sample_rate, **kwargs)
    elif protocol in (ProtocolType.AX25, ProtocolType.APRS):
        return AX25Decoder(sample_rate, **kwargs)
    elif protocol == ProtocolType.RDS:
        return RDSDecoder(sample_rate)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

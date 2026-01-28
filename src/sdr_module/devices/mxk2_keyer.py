"""
MX-K2 CW Keyer device driver.

Provides interface for the MX-K2 CW keyer via USB serial connection.
The MX-K2 is a standalone CW keyer with iambic paddle support,
adjustable speed, and built-in memory keying.

Specifications:
    - Interface: USB serial (virtual COM port)
    - Baud rate: 1200 (default)
    - Speed range: 5-50 WPM
    - Modes: Iambic A, Iambic B, Ultimatic, Bug, Straight key
    - Sidetone: 300-1200 Hz
    - Keying output: PTT and CW key lines
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

import numpy as np

from .base import (
    DeviceCapability,
    DeviceInfo,
    DeviceSpec,
    SDRDevice,
)

logger = logging.getLogger(__name__)


class PaddleMode(Enum):
    """Keyer paddle modes."""

    IAMBIC_A = auto()  # Iambic mode A (alternating, completes element)
    IAMBIC_B = auto()  # Iambic mode B (alternating, adds opposite element)
    ULTIMATIC = auto()  # Ultimatic mode (last paddle wins)
    BUG = auto()  # Semi-automatic bug mode (auto dits, manual dahs)
    STRAIGHT = auto()  # Straight key mode (manual timing)


class KeyerState(Enum):
    """Keyer operational state."""

    IDLE = auto()  # Not keying
    KEYING = auto()  # Currently sending
    MEMORY_PLAYBACK = auto()  # Playing back stored message


@dataclass
class KeyerStatus:
    """Current keyer status."""

    is_connected: bool = False
    is_keying: bool = False
    wpm: int = 20
    sidetone_freq: int = 700
    paddle_mode: PaddleMode = PaddleMode.IAMBIC_B
    ptt_active: bool = False
    key_down: bool = False
    memory_slot: int = 0
    state: KeyerState = KeyerState.IDLE
    firmware_version: str = ""


@dataclass
class MXK2Config:
    """Configuration for MX-K2 keyer."""

    port: str = ""  # Serial port (e.g., "/dev/ttyUSB0" or "COM3")
    baud_rate: int = 1200  # Default baud rate for MX-K2
    wpm: int = 20  # Words per minute (5-50)
    sidetone_freq: int = 700  # Sidetone frequency in Hz (300-1200)
    sidetone_enabled: bool = True
    paddle_mode: PaddleMode = PaddleMode.IAMBIC_B
    paddle_swap: bool = False  # Swap dit/dah paddles
    weight: int = 50  # Dit/dah weight (25-75, 50 = standard 1:3)
    dah_to_dit_ratio: float = 3.0  # Standard Morse timing
    ptt_lead_time_ms: int = 50  # PTT lead time before keying
    ptt_tail_time_ms: int = 100  # PTT hang time after keying
    auto_space: bool = False  # Automatic inter-character spacing
    command_mode: bool = False  # Enable command input via paddles
    memory_slots: List[str] = field(default_factory=lambda: [""] * 4)


# MX-K2 device specifications
MXK2_SPEC = DeviceSpec(
    freq_min=0,  # Not applicable for keyer
    freq_max=0,
    sample_rate_min=0,
    sample_rate_max=0,
    bandwidth_max=0,
    adc_bits=0,
    gain_min=0,
    gain_max=0,
    max_input_power=0,
    tx_power_min=None,
    tx_power_max=None,
)


# MX-K2 Protocol Commands
class MXK2Command:
    """MX-K2 serial protocol commands."""

    # Speed commands
    SET_SPEED = b"S"  # S<wpm> - Set speed (5-50)
    GET_SPEED = b"s"  # Query current speed

    # Sidetone commands
    SET_SIDETONE = b"T"  # T<freq> - Set sidetone frequency
    SIDETONE_ON = b"O"  # Enable sidetone
    SIDETONE_OFF = b"o"  # Disable sidetone

    # Mode commands
    SET_IAMBIC_A = b"A"  # Set Iambic A mode
    SET_IAMBIC_B = b"B"  # Set Iambic B mode
    SET_ULTIMATIC = b"U"  # Set Ultimatic mode
    SET_BUG = b"G"  # Set Bug mode
    SET_STRAIGHT = b"K"  # Set straight key mode

    # Keying commands
    KEY_DOWN = b"D"  # Key down (start keying)
    KEY_UP = b"U"  # Key up (stop keying)
    SEND_TEXT = b"W"  # W<text> - Send text as CW
    ABORT = b"X"  # Abort current transmission

    # Memory commands
    PLAY_MEMORY = b"M"  # M<slot> - Play memory slot (1-4)
    STORE_MEMORY = b"P"  # P<slot><text> - Store text in memory

    # PTT commands
    PTT_ON = b"+"  # Assert PTT
    PTT_OFF = b"-"  # Release PTT

    # Configuration commands
    SET_WEIGHT = b"E"  # E<weight> - Set weight (25-75)
    SWAP_PADDLES = b"R"  # Toggle paddle swap
    AUTO_SPACE = b"Z"  # Toggle auto-space

    # Query commands
    QUERY_STATUS = b"?"  # Query full status
    QUERY_VERSION = b"V"  # Query firmware version

    # Acknowledgment
    ACK = b"\x06"  # Command acknowledged
    NAK = b"\x15"  # Command not acknowledged


class MXK2Keyer(SDRDevice):
    """
    MX-K2 CW Keyer device driver.

    Provides serial communication with the MX-K2 keyer for:
    - Setting and reading keyer speed (WPM)
    - Sending CW text
    - Controlling PTT
    - Reading paddle input
    - Memory keyer functions

    Example:
        >>> keyer = MXK2Keyer()
        >>> keyer.open(port="/dev/ttyUSB0")
        >>> keyer.set_wpm(25)
        >>> keyer.send_text("CQ CQ CQ DE W1AW")
        >>> keyer.close()
    """

    def __init__(self, config: Optional[MXK2Config] = None):
        super().__init__()
        self._config = config or MXK2Config()
        self._serial = None
        self._spec = MXK2_SPEC
        self._status = KeyerStatus()
        self._read_thread: Optional[threading.Thread] = None
        self._stop_read = threading.Event()
        self._response_buffer = bytearray()
        self._lock = threading.Lock()
        self._on_key_change: Optional[Callable[[bool], None]] = None
        self._on_ptt_change: Optional[Callable[[bool], None]] = None

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """
        List available MX-K2 keyer devices.

        Scans serial ports for MX-K2 devices by attempting connection
        and querying the firmware version.

        Returns:
            List of detected MX-K2 devices
        """
        devices = []
        try:
            import serial.tools.list_ports

            ports = serial.tools.list_ports.comports()
            for port in ports:
                # Check for known USB-serial adapters commonly used with MX-K2
                # MX-K2 typically uses a Prolific or FTDI USB-serial chip
                description = port.description.lower()
                if any(
                    x in description
                    for x in ["usb", "serial", "prolific", "ftdi", "ch340"]
                ):
                    # Try to identify as MX-K2
                    try:
                        import serial

                        ser = serial.Serial(port.device, 1200, timeout=0.5)
                        ser.write(MXK2Command.QUERY_VERSION)
                        time.sleep(0.1)
                        response = ser.read(100)
                        ser.close()

                        # Check for MX-K2 identifier in response
                        if b"MX-K2" in response or b"K2" in response:
                            version = response.decode("ascii", errors="ignore").strip()
                            devices.append(
                                DeviceInfo(
                                    name="MX-K2 CW Keyer",
                                    serial=port.serial_number or port.device,
                                    manufacturer="MFJ Enterprises",
                                    product="MX-K2",
                                    index=len(devices),
                                    capabilities=[
                                        DeviceCapability.TX,  # Keyer can transmit
                                        DeviceCapability.HALF_DUPLEX,
                                    ],
                                )
                            )
                            logger.info(
                                f"Found MX-K2 on {port.device}: {version}"
                            )
                    except Exception as e:
                        logger.debug(f"Port {port.device} not MX-K2: {e}")
        except ImportError:
            logger.warning("pyserial not installed - cannot scan for keyers")
        except Exception as e:
            logger.error(f"Error scanning for MX-K2 devices: {e}")

        return devices

    @staticmethod
    def find_keyer_port() -> Optional[str]:
        """
        Auto-detect the MX-K2 serial port.

        Returns:
            Serial port path if found, None otherwise
        """
        devices = MXK2Keyer.list_devices()
        if devices:
            return devices[0].serial
        return None

    def open(self, index: int = 0, port: Optional[str] = None) -> bool:
        """
        Open connection to the MX-K2 keyer.

        Args:
            index: Device index (if multiple keyers)
            port: Serial port path (overrides auto-detection)

        Returns:
            True if connection successful
        """
        if self._is_open:
            logger.warning("Keyer already open")
            return True

        try:
            import serial

            # Determine port to use
            target_port = port or self._config.port
            if not target_port:
                target_port = self.find_keyer_port()
                if not target_port:
                    logger.error("No MX-K2 keyer found")
                    return False

            # Open serial connection
            self._serial = serial.Serial(
                port=target_port,
                baudrate=self._config.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=1.0,
            )

            self._is_open = True
            self._status.is_connected = True

            # Query device info
            self._query_version()

            # Set device info
            self._info = DeviceInfo(
                name="MX-K2 CW Keyer",
                serial=target_port,
                manufacturer="MFJ Enterprises",
                product="MX-K2",
                index=index,
                capabilities=[
                    DeviceCapability.TX,
                    DeviceCapability.HALF_DUPLEX,
                ],
            )

            # Apply initial configuration
            self._apply_config()

            # Start read thread for status updates
            self._start_read_thread()

            logger.info(f"Opened MX-K2 keyer on {target_port}")
            return True

        except ImportError:
            logger.error("pyserial not installed. Install with: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"Failed to open MX-K2: {e}")
            # Clean up serial port on failure to prevent resource leak
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception as e:
                    logger.debug(f"Error closing serial port during cleanup: {e}")
                self._serial = None
            self._is_open = False
            self._status.is_connected = False
            return False

    def close(self) -> None:
        """Close connection to the keyer."""
        self._stop_read.set()

        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)

        if self._serial:
            try:
                # Ensure key is released
                self._send_command(MXK2Command.KEY_UP)
                self._send_command(MXK2Command.PTT_OFF)
                self._serial.close()
            except Exception as e:
                logger.error(f"Error closing keyer: {e}")
            finally:
                self._serial = None

        self._is_open = False
        self._status.is_connected = False
        logger.info("MX-K2 keyer closed")

    def _send_command(self, cmd: bytes, data: bytes = b"") -> bool:
        """Send a command to the keyer."""
        if not self._is_open or not self._serial:
            return False

        try:
            with self._lock:
                self._serial.write(cmd + data + b"\r")
                self._serial.flush()
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    def _read_response(self, timeout: float = 0.5, max_size: int = 1024) -> bytes:
        """Read response from the keyer.

        Args:
            timeout: Read timeout in seconds
            max_size: Maximum bytes to read (prevents unbounded reads from malicious devices)
        """
        if not self._is_open or not self._serial:
            return b""

        try:
            with self._lock:
                self._serial.timeout = timeout
                response = self._serial.read_until(b"\r", size=max_size)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to read response: {e}")
            return b""

    def _query_version(self) -> None:
        """Query and store firmware version."""
        self._send_command(MXK2Command.QUERY_VERSION)
        response = self._read_response()
        if response:
            self._status.firmware_version = response.decode("ascii", errors="ignore")

    def _apply_config(self) -> None:
        """Apply current configuration to the keyer."""
        self.set_wpm(self._config.wpm)
        self.set_sidetone(self._config.sidetone_freq, self._config.sidetone_enabled)
        self.set_paddle_mode(self._config.paddle_mode)
        self.set_weight(self._config.weight)
        if self._config.paddle_swap:
            self.swap_paddles()

    def _start_read_thread(self) -> None:
        """Start background thread for reading keyer status."""
        self._stop_read.clear()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def _read_loop(self) -> None:
        """Background loop to read keyer status updates."""
        while not self._stop_read.is_set():
            try:
                if self._serial and self._serial.in_waiting > 0:
                    with self._lock:
                        data = self._serial.read(self._serial.in_waiting)

                    # Parse status updates
                    self._parse_status_update(data)
            except Exception as e:
                if not self._stop_read.is_set():
                    logger.debug(f"Read loop error: {e}")

            time.sleep(0.01)  # Small delay to prevent busy waiting

    def _parse_status_update(self, data: bytes) -> None:
        """Parse asynchronous status updates from the keyer."""
        # Status update format varies by implementation
        # Common patterns:
        # K1 = key down, K0 = key up
        # P1 = PTT on, P0 = PTT off
        for byte in data:
            char = chr(byte)
            if char == "K":
                # Key state follows
                pass
            elif char == "P":
                # PTT state follows
                pass

    # =========================================================================
    # SDRDevice Interface Implementation (required abstract methods)
    # =========================================================================

    def set_frequency(self, freq_hz: float) -> bool:
        """Not applicable for keyer - returns True for compatibility."""
        logger.debug("set_frequency called on keyer (no-op)")
        return True

    def set_sample_rate(self, rate_hz: float) -> bool:
        """Not applicable for keyer - returns True for compatibility."""
        logger.debug("set_sample_rate called on keyer (no-op)")
        return True

    def set_bandwidth(self, bw_hz: float) -> bool:
        """Not applicable for keyer - returns True for compatibility."""
        logger.debug("set_bandwidth called on keyer (no-op)")
        return True

    def set_gain(self, gain_db: float) -> bool:
        """Not applicable for keyer - returns True for compatibility."""
        logger.debug("set_gain called on keyer (no-op)")
        return True

    def set_gain_mode(self, auto: bool) -> bool:
        """Not applicable for keyer - returns True for compatibility."""
        logger.debug("set_gain_mode called on keyer (no-op)")
        return True

    def start_rx(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """Not applicable for keyer - returns False."""
        logger.warning("MX-K2 keyer does not support RX")
        return False

    def stop_rx(self) -> bool:
        """Not applicable for keyer - returns True."""
        return True

    # =========================================================================
    # Keyer-Specific Methods
    # =========================================================================

    def set_wpm(self, wpm: int) -> bool:
        """
        Set keyer speed in words per minute.

        Args:
            wpm: Speed in WPM (5-50)

        Returns:
            True if successful
        """
        wpm = max(5, min(50, wpm))
        if self._send_command(MXK2Command.SET_SPEED, str(wpm).encode()):
            self._config.wpm = wpm
            self._status.wpm = wpm
            logger.debug(f"Set keyer speed to {wpm} WPM")
            return True
        return False

    def get_wpm(self) -> int:
        """Get current keyer speed."""
        return self._status.wpm

    def set_sidetone(self, freq_hz: int, enabled: bool = True) -> bool:
        """
        Configure sidetone.

        Args:
            freq_hz: Sidetone frequency (300-1200 Hz)
            enabled: Enable or disable sidetone

        Returns:
            True if successful
        """
        freq_hz = max(300, min(1200, freq_hz))

        success = True
        if not self._send_command(MXK2Command.SET_SIDETONE, str(freq_hz).encode()):
            success = False

        cmd = MXK2Command.SIDETONE_ON if enabled else MXK2Command.SIDETONE_OFF
        if not self._send_command(cmd):
            success = False

        if success:
            self._config.sidetone_freq = freq_hz
            self._config.sidetone_enabled = enabled
            self._status.sidetone_freq = freq_hz
            logger.debug(f"Set sidetone to {freq_hz} Hz, enabled={enabled}")

        return success

    def set_paddle_mode(self, mode: PaddleMode) -> bool:
        """
        Set paddle operating mode.

        Args:
            mode: Paddle mode (IAMBIC_A, IAMBIC_B, ULTIMATIC, BUG, STRAIGHT)

        Returns:
            True if successful
        """
        mode_commands = {
            PaddleMode.IAMBIC_A: MXK2Command.SET_IAMBIC_A,
            PaddleMode.IAMBIC_B: MXK2Command.SET_IAMBIC_B,
            PaddleMode.ULTIMATIC: MXK2Command.SET_ULTIMATIC,
            PaddleMode.BUG: MXK2Command.SET_BUG,
            PaddleMode.STRAIGHT: MXK2Command.SET_STRAIGHT,
        }

        cmd = mode_commands.get(mode)
        if cmd and self._send_command(cmd):
            self._config.paddle_mode = mode
            self._status.paddle_mode = mode
            logger.debug(f"Set paddle mode to {mode.name}")
            return True
        return False

    def set_weight(self, weight: int) -> bool:
        """
        Set dit/dah weight ratio.

        Args:
            weight: Weight value (25-75, 50 = standard 1:3 ratio)

        Returns:
            True if successful
        """
        weight = max(25, min(75, weight))
        if self._send_command(MXK2Command.SET_WEIGHT, str(weight).encode()):
            self._config.weight = weight
            logger.debug(f"Set weight to {weight}")
            return True
        return False

    def swap_paddles(self) -> bool:
        """Toggle paddle swap (dit/dah reversal)."""
        if self._send_command(MXK2Command.SWAP_PADDLES):
            self._config.paddle_swap = not self._config.paddle_swap
            logger.debug(f"Paddle swap: {self._config.paddle_swap}")
            return True
        return False

    def send_text(self, text: str) -> bool:
        """
        Send text as CW.

        The keyer will convert the text to Morse code and transmit it.

        Args:
            text: Text to send (A-Z, 0-9, and common punctuation)

        Returns:
            True if command sent successfully
        """
        if not text:
            return False

        # Clean text - only valid Morse characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?/-=")
        original_text = text.upper()
        filtered_text = "".join(c for c in original_text if c in valid_chars)

        # Warn user if characters were filtered
        if filtered_text != original_text:
            removed = set(original_text) - set(filtered_text)
            logger.warning(
                f"Invalid Morse characters removed from text: {removed}. "
                f"Sending: '{filtered_text}'"
            )

        if not filtered_text:
            logger.warning("No valid Morse characters in text after filtering")
            return False

        if self._send_command(MXK2Command.SEND_TEXT, filtered_text.encode("ascii")):
            self._status.is_keying = True
            self._status.state = KeyerState.KEYING
            logger.info(f"Sending CW: {filtered_text}")
            return True
        return False

    def abort(self) -> bool:
        """Abort current transmission."""
        if self._send_command(MXK2Command.ABORT):
            self._status.is_keying = False
            self._status.state = KeyerState.IDLE
            logger.info("Transmission aborted")
            return True
        return False

    def key_down(self) -> bool:
        """Assert key line (start keying)."""
        if self._send_command(MXK2Command.KEY_DOWN):
            self._status.key_down = True
            if self._on_key_change:
                self._on_key_change(True)
            return True
        return False

    def key_up(self) -> bool:
        """Release key line (stop keying)."""
        if self._send_command(MXK2Command.KEY_UP):
            self._status.key_down = False
            if self._on_key_change:
                self._on_key_change(False)
            return True
        return False

    def ptt_on(self) -> bool:
        """Assert PTT line."""
        if self._send_command(MXK2Command.PTT_ON):
            self._status.ptt_active = True
            if self._on_ptt_change:
                self._on_ptt_change(True)
            return True
        return False

    def ptt_off(self) -> bool:
        """Release PTT line."""
        if self._send_command(MXK2Command.PTT_OFF):
            self._status.ptt_active = False
            if self._on_ptt_change:
                self._on_ptt_change(False)
            return True
        return False

    def play_memory(self, slot: int) -> bool:
        """
        Play a stored memory message.

        Args:
            slot: Memory slot number (1-4)

        Returns:
            True if command sent successfully
        """
        if not 1 <= slot <= 4:
            logger.error(f"Invalid memory slot: {slot}")
            return False

        if self._send_command(MXK2Command.PLAY_MEMORY, str(slot).encode()):
            self._status.memory_slot = slot
            self._status.state = KeyerState.MEMORY_PLAYBACK
            logger.info(f"Playing memory slot {slot}")
            return True
        return False

    def store_memory(self, slot: int, text: str) -> bool:
        """
        Store text in a memory slot.

        Args:
            slot: Memory slot number (1-4)
            text: Text to store

        Returns:
            True if successful
        """
        if not 1 <= slot <= 4:
            logger.error(f"Invalid memory slot: {slot}")
            return False

        # Sanitize text - only valid Morse characters, limit length
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?/-=")
        original_text = text.upper()
        filtered_text = "".join(c for c in original_text if c in valid_chars)[:50]

        if filtered_text != original_text[:50]:
            removed = set(original_text) - set(filtered_text) - valid_chars
            if removed:
                logger.warning(f"Invalid Morse characters removed from memory text: {removed}")

        data = f"{slot}{filtered_text}".encode("ascii")

        if self._send_command(MXK2Command.STORE_MEMORY, data):
            self._config.memory_slots[slot - 1] = filtered_text
            logger.info(f"Stored in memory slot {slot}: {filtered_text}")
            return True
        return False

    def set_on_key_change(self, callback: Callable[[bool], None]) -> None:
        """
        Set callback for key state changes.

        Args:
            callback: Function called with True (key down) or False (key up)
        """
        self._on_key_change = callback

    def set_on_ptt_change(self, callback: Callable[[bool], None]) -> None:
        """
        Set callback for PTT state changes.

        Args:
            callback: Function called with True (PTT on) or False (PTT off)
        """
        self._on_ptt_change = callback

    def get_status(self) -> KeyerStatus:
        """Get current keyer status."""
        return self._status

    def get_config(self) -> MXK2Config:
        """Get current keyer configuration."""
        return self._config

    @property
    def is_keying(self) -> bool:
        """Check if keyer is currently sending."""
        return self._status.is_keying

    def __repr__(self) -> str:
        if self._info:
            return f"<MXK2Keyer {self._info.serial} @ {self._status.wpm} WPM>"
        return "<MXK2Keyer (not connected)>"


# Convenience function for quick keyer setup
def create_keyer(
    port: Optional[str] = None, wpm: int = 20, sidetone_hz: int = 700
) -> Optional[MXK2Keyer]:
    """
    Create and configure an MX-K2 keyer.

    Args:
        port: Serial port (auto-detect if None)
        wpm: Initial speed in WPM
        sidetone_hz: Sidetone frequency

    Returns:
        Configured MXK2Keyer instance, or None if not found
    """
    config = MXK2Config(port=port or "", wpm=wpm, sidetone_freq=sidetone_hz)
    keyer = MXK2Keyer(config)

    if keyer.open(port=port):
        return keyer

    return None


__all__ = [
    "MXK2Keyer",
    "MXK2Config",
    "MXK2Command",
    "PaddleMode",
    "KeyerState",
    "KeyerStatus",
    "create_keyer",
]

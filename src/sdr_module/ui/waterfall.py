"""
Waterfall display with packet highlighting.

Provides time-frequency visualization with:
- Scrolling waterfall spectrogram
- Protocol-based packet highlighting
- Color-coded overlays for detected signals
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import colorsys


class ColorMap(Enum):
    """Waterfall color maps."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    TURBO = "turbo"
    GRAYSCALE = "grayscale"
    CLASSIC = "classic"  # Blue-cyan-green-yellow-red


@dataclass
class PacketHighlight:
    """Highlighted packet region on waterfall."""
    time_start: int         # Start row (time index)
    time_end: int           # End row (time index)
    freq_start: int         # Start column (frequency bin)
    freq_end: int           # End column (frequency bin)
    protocol: str           # Protocol name
    color: Tuple[int, int, int, int]  # RGBA color
    label: str = ""         # Optional label text
    confidence: float = 1.0 # Detection confidence
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProtocolColorScheme:
    """Color scheme for protocol highlighting."""
    name: str
    color: Tuple[int, int, int]  # RGB
    alpha: int = 180             # Transparency (0-255)
    border_color: Tuple[int, int, int] = (255, 255, 255)
    border_width: int = 2


# Default protocol color schemes
PROTOCOL_COLORS: Dict[str, ProtocolColorScheme] = {
    # ISM Band protocols
    "ook": ProtocolColorScheme("OOK", (255, 165, 0)),      # Orange
    "ask": ProtocolColorScheme("ASK", (255, 140, 0)),      # Dark orange
    "fsk": ProtocolColorScheme("FSK", (0, 191, 255)),      # Deep sky blue
    "gfsk": ProtocolColorScheme("GFSK", (30, 144, 255)),   # Dodger blue

    # Amateur radio
    "ax25": ProtocolColorScheme("AX.25", (50, 205, 50)),   # Lime green
    "aprs": ProtocolColorScheme("APRS", (34, 139, 34)),    # Forest green
    "ft8": ProtocolColorScheme("FT8", (0, 255, 127)),      # Spring green
    "wspr": ProtocolColorScheme("WSPR", (0, 250, 154)),    # Medium spring green

    # Aviation
    "adsb": ProtocolColorScheme("ADS-B", (255, 0, 0)),     # Red
    "acars": ProtocolColorScheme("ACARS", (220, 20, 60)),  # Crimson
    "vdl2": ProtocolColorScheme("VDL2", (178, 34, 34)),    # Firebrick

    # Paging
    "pocsag": ProtocolColorScheme("POCSAG", (255, 20, 147)), # Deep pink
    "flex": ProtocolColorScheme("FLEX", (255, 105, 180)),    # Hot pink

    # Trunking
    "p25": ProtocolColorScheme("P25", (138, 43, 226)),     # Blue violet
    "dmr": ProtocolColorScheme("DMR", (148, 0, 211)),      # Dark violet
    "tetra": ProtocolColorScheme("TETRA", (186, 85, 211)), # Medium orchid
    "nxdn": ProtocolColorScheme("NXDN", (153, 50, 204)),   # Dark orchid

    # IoT
    "lora": ProtocolColorScheme("LoRa", (0, 255, 255)),    # Cyan
    "zigbee": ProtocolColorScheme("Zigbee", (0, 206, 209)), # Dark turquoise
    "zwave": ProtocolColorScheme("Z-Wave", (64, 224, 208)), # Turquoise
    "bluetooth": ProtocolColorScheme("Bluetooth", (0, 0, 255)), # Blue

    # Broadcast
    "rds": ProtocolColorScheme("RDS", (255, 215, 0)),      # Gold
    "dab": ProtocolColorScheme("DAB", (255, 223, 0)),      # Golden yellow

    # Generic
    "unknown": ProtocolColorScheme("Unknown", (128, 128, 128)), # Gray
    "analog": ProtocolColorScheme("Analog", (255, 255, 0)),     # Yellow
    "digital": ProtocolColorScheme("Digital", (0, 255, 0)),     # Green
    "noise": ProtocolColorScheme("Noise", (64, 64, 64)),        # Dark gray
}


class WaterfallDisplay:
    """
    Waterfall spectrogram display with packet highlighting.

    Maintains a scrolling history of spectrum data and overlays
    detected packet regions with protocol-specific colors.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 512,
        colormap: ColorMap = ColorMap.TURBO,
        min_db: float = -100,
        max_db: float = -20
    ):
        """
        Initialize waterfall display.

        Args:
            width: Display width in pixels (frequency bins)
            height: Display height in pixels (time history)
            colormap: Color map for spectrum display
            min_db: Minimum power level (dB) for color mapping
            max_db: Maximum power level (dB) for color mapping
        """
        self._width = width
        self._height = height
        self._colormap = colormap
        self._min_db = min_db
        self._max_db = max_db

        # Waterfall data buffer (newest at bottom)
        self._data = np.zeros((height, width), dtype=np.float32)
        self._data.fill(min_db)

        # RGB image buffer
        self._image = np.zeros((height, width, 4), dtype=np.uint8)

        # Packet highlights
        self._highlights: List[PacketHighlight] = []
        self._max_highlights = 1000  # Limit stored highlights

        # Color lookup table
        self._color_lut = self._create_color_lut(colormap)

        # Current time index (row)
        self._time_index = 0

        # Frequency axis info
        self._center_freq = 0.0
        self._sample_rate = 1.0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def image(self) -> np.ndarray:
        """Get current waterfall image (RGBA)."""
        return self._image.copy()

    @property
    def data(self) -> np.ndarray:
        """Get raw waterfall data (dB values)."""
        return self._data.copy()

    @property
    def highlights(self) -> List[PacketHighlight]:
        """Get list of packet highlights."""
        return self._highlights.copy()

    def set_frequency_range(self, center_freq: float, sample_rate: float) -> None:
        """Set frequency axis parameters."""
        self._center_freq = center_freq
        self._sample_rate = sample_rate

    def set_dynamic_range(self, min_db: float, max_db: float) -> None:
        """Set power level range for color mapping."""
        self._min_db = min_db
        self._max_db = max_db
        self._update_image()

    def set_colormap(self, colormap: ColorMap) -> None:
        """Change color map."""
        self._colormap = colormap
        self._color_lut = self._create_color_lut(colormap)
        self._update_image()

    def add_spectrum_line(self, power_db: np.ndarray) -> int:
        """
        Add a new spectrum line to the waterfall.

        Args:
            power_db: Power spectrum in dB (length should match width)

        Returns:
            Time index of added line
        """
        # Resample if necessary
        if len(power_db) != self._width:
            power_db = np.interp(
                np.linspace(0, len(power_db) - 1, self._width),
                np.arange(len(power_db)),
                power_db
            )

        # Scroll up (oldest data at top)
        self._data[:-1] = self._data[1:]
        self._data[-1] = power_db

        # Update image
        self._update_last_line()

        self._time_index += 1

        # Scroll highlights
        self._scroll_highlights()

        return self._time_index

    def add_packet_highlight(
        self,
        freq_start_hz: float,
        freq_end_hz: float,
        duration_lines: int,
        protocol: str,
        label: str = "",
        confidence: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> PacketHighlight:
        """
        Add a packet highlight at the current position.

        Args:
            freq_start_hz: Start frequency in Hz
            freq_end_hz: End frequency in Hz
            duration_lines: Duration in waterfall lines
            protocol: Protocol name for color coding
            label: Optional display label
            confidence: Detection confidence (0-1)
            metadata: Optional metadata dict

        Returns:
            Created PacketHighlight object
        """
        # Convert frequency to bins
        freq_min = self._center_freq - self._sample_rate / 2
        self._center_freq + self._sample_rate / 2
        hz_per_bin = self._sample_rate / self._width

        freq_start_bin = int((freq_start_hz - freq_min) / hz_per_bin)
        freq_end_bin = int((freq_end_hz - freq_min) / hz_per_bin)

        # Clamp to valid range
        freq_start_bin = max(0, min(self._width - 1, freq_start_bin))
        freq_end_bin = max(0, min(self._width - 1, freq_end_bin))

        # Get color for protocol
        color = self._get_protocol_color(protocol, confidence)

        highlight = PacketHighlight(
            time_start=self._height - duration_lines,
            time_end=self._height - 1,
            freq_start=freq_start_bin,
            freq_end=freq_end_bin,
            protocol=protocol,
            color=color,
            label=label,
            confidence=confidence,
            metadata=metadata or {}
        )

        self._highlights.append(highlight)

        # Limit stored highlights
        if len(self._highlights) > self._max_highlights:
            self._highlights = self._highlights[-self._max_highlights:]

        # Draw highlight on image
        self._draw_highlight(highlight)

        return highlight

    def add_highlight_at_position(
        self,
        time_start: int,
        time_end: int,
        freq_start_bin: int,
        freq_end_bin: int,
        protocol: str,
        label: str = "",
        confidence: float = 1.0
    ) -> PacketHighlight:
        """
        Add highlight at specific position (in display coordinates).

        Args:
            time_start: Start row
            time_end: End row
            freq_start_bin: Start frequency bin
            freq_end_bin: End frequency bin
            protocol: Protocol name
            label: Display label
            confidence: Detection confidence

        Returns:
            Created PacketHighlight
        """
        color = self._get_protocol_color(protocol, confidence)

        highlight = PacketHighlight(
            time_start=time_start,
            time_end=time_end,
            freq_start=freq_start_bin,
            freq_end=freq_end_bin,
            protocol=protocol,
            color=color,
            label=label,
            confidence=confidence
        )

        self._highlights.append(highlight)
        self._draw_highlight(highlight)

        return highlight

    def clear_highlights(self) -> None:
        """Clear all packet highlights."""
        self._highlights.clear()
        self._update_image()

    def get_highlights_at(self, x: int, y: int) -> List[PacketHighlight]:
        """Get highlights at a specific position."""
        hits = []
        for h in self._highlights:
            if (h.freq_start <= x <= h.freq_end and
                h.time_start <= y <= h.time_end):
                hits.append(h)
        return hits

    def _get_protocol_color(
        self,
        protocol: str,
        confidence: float = 1.0
    ) -> Tuple[int, int, int, int]:
        """Get RGBA color for protocol."""
        protocol_lower = protocol.lower()
        scheme = PROTOCOL_COLORS.get(
            protocol_lower,
            PROTOCOL_COLORS["unknown"]
        )

        # Adjust alpha by confidence
        alpha = int(scheme.alpha * confidence)

        return (scheme.color[0], scheme.color[1], scheme.color[2], alpha)

    def _create_color_lut(self, colormap: ColorMap) -> np.ndarray:
        """Create color lookup table (256 entries)."""
        lut = np.zeros((256, 4), dtype=np.uint8)

        for i in range(256):
            t = i / 255.0

            if colormap == ColorMap.GRAYSCALE:
                r = g = b = int(t * 255)

            elif colormap == ColorMap.CLASSIC:
                # Blue -> Cyan -> Green -> Yellow -> Red
                if t < 0.25:
                    r, g, b = 0, int(t * 4 * 255), 255
                elif t < 0.5:
                    r, g, b = 0, 255, int((0.5 - t) * 4 * 255)
                elif t < 0.75:
                    r, g, b = int((t - 0.5) * 4 * 255), 255, 0
                else:
                    r, g, b = 255, int((1.0 - t) * 4 * 255), 0

            elif colormap == ColorMap.VIRIDIS:
                # Approximation of viridis
                r = int((0.267 + 0.329 * t + 2.211 * t**2 - 1.807 * t**3) * 255)
                g = int((0.004 + 1.260 * t - 0.637 * t**2 + 0.373 * t**3) * 255)
                b = int((0.329 + 1.421 * t - 2.682 * t**2 + 1.932 * t**3) * 255)
                r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

            elif colormap == ColorMap.PLASMA:
                # Approximation of plasma
                r = int((0.050 + 2.810 * t - 2.251 * t**2 + 0.667 * t**3) * 255)
                g = int((0.030 + 0.115 * t + 2.292 * t**2 - 1.433 * t**3) * 255)
                b = int((0.530 + 1.622 * t - 4.115 * t**2 + 2.963 * t**3) * 255)
                r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

            elif colormap == ColorMap.TURBO:
                # Approximation of turbo
                r = int((0.135 + 4.243 * t - 14.72 * t**2 + 23.07 * t**3 - 17.66 * t**4 + 5.055 * t**5) * 255)
                g = int((0.091 + 3.109 * t - 5.437 * t**2 + 3.093 * t**3 - 0.855 * t**4) * 255)
                b = int((0.107 + 5.637 * t - 19.14 * t**2 + 27.42 * t**3 - 17.66 * t**4 + 4.709 * t**5) * 255)
                r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

            else:  # INFERNO, MAGMA, etc - use HSV approximation
                h = (1.0 - t) * 0.7  # Hue from red to blue
                s = 0.9
                v = 0.1 + 0.9 * t
                r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]

            lut[i] = [r, g, b, 255]

        return lut

    def _update_image(self) -> None:
        """Update entire image from data."""
        # Normalize data to 0-255
        normalized = np.clip(
            (self._data - self._min_db) / (self._max_db - self._min_db),
            0, 1
        )
        indices = (normalized * 255).astype(np.uint8)

        # Apply color map
        self._image = self._color_lut[indices]

        # Redraw all highlights
        for highlight in self._highlights:
            self._draw_highlight(highlight)

    def _update_last_line(self) -> None:
        """Update only the last line of the image."""
        # Scroll image
        self._image[:-1] = self._image[1:]

        # Update last line
        normalized = np.clip(
            (self._data[-1] - self._min_db) / (self._max_db - self._min_db),
            0, 1
        )
        indices = (normalized * 255).astype(np.uint8)
        self._image[-1] = self._color_lut[indices]

    def _scroll_highlights(self) -> None:
        """Scroll highlights up and remove old ones."""
        active_highlights = []

        for h in self._highlights:
            # Scroll up
            h.time_start -= 1
            h.time_end -= 1

            # Keep if still visible
            if h.time_end >= 0:
                active_highlights.append(h)

        self._highlights = active_highlights

    def _draw_highlight(self, highlight: PacketHighlight) -> None:
        """Draw a highlight rectangle on the image."""
        t1 = max(0, highlight.time_start)
        t2 = min(self._height - 1, highlight.time_end)
        f1 = max(0, highlight.freq_start)
        f2 = min(self._width - 1, highlight.freq_end)

        if t1 > t2 or f1 > f2:
            return

        r, g, b, a = highlight.color
        alpha = a / 255.0

        # Blend highlight with existing image
        for t in range(t1, t2 + 1):
            for f in range(f1, f2 + 1):
                # Border
                is_border = (t == t1 or t == t2 or f == f1 or f == f2)
                if is_border:
                    self._image[t, f] = [255, 255, 255, 255]
                else:
                    # Alpha blend
                    old = self._image[t, f].astype(np.float32)
                    new = np.array([r, g, b, 255], dtype=np.float32)
                    blended = old * (1 - alpha) + new * alpha
                    self._image[t, f] = blended.astype(np.uint8)

    def get_frequency_at_bin(self, bin_index: int) -> float:
        """Convert bin index to frequency in Hz."""
        freq_min = self._center_freq - self._sample_rate / 2
        hz_per_bin = self._sample_rate / self._width
        return freq_min + bin_index * hz_per_bin

    def get_bin_at_frequency(self, freq_hz: float) -> int:
        """Convert frequency in Hz to bin index."""
        freq_min = self._center_freq - self._sample_rate / 2
        hz_per_bin = self._sample_rate / self._width
        return int((freq_hz - freq_min) / hz_per_bin)


def get_protocol_color(protocol: str) -> Tuple[int, int, int]:
    """Get RGB color for a protocol."""
    scheme = PROTOCOL_COLORS.get(
        protocol.lower(),
        PROTOCOL_COLORS["unknown"]
    )
    return scheme.color


def list_protocol_colors() -> Dict[str, Tuple[int, int, int]]:
    """Get dictionary of all protocol colors."""
    return {name: scheme.color for name, scheme in PROTOCOL_COLORS.items()}


def register_protocol_color(
    protocol: str,
    color: Tuple[int, int, int],
    alpha: int = 180
) -> None:
    """Register a custom protocol color."""
    PROTOCOL_COLORS[protocol.lower()] = ProtocolColorScheme(
        name=protocol,
        color=color,
        alpha=alpha
    )

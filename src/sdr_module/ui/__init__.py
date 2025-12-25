"""
UI module - User interface components.

Provides visualization components for SDR applications:
- Waterfall display with packet highlighting
- Protocol color coding
- Real-time signal detection overlay
"""

from .waterfall import (
    WaterfallDisplay,
    PacketHighlight,
    ColorMap,
    ProtocolColorScheme,
    PROTOCOL_COLORS,
    get_protocol_color,
    list_protocol_colors,
    register_protocol_color,
)

from .packet_highlighter import (
    PacketHighlighter,
    DetectionMode,
    DetectionConfig,
    DetectedPacket,
    LivePacketDisplay,
)

__all__ = [
    # Waterfall
    "WaterfallDisplay",
    "PacketHighlight",
    "ColorMap",
    "ProtocolColorScheme",
    "PROTOCOL_COLORS",
    "get_protocol_color",
    "list_protocol_colors",
    "register_protocol_color",
    # Packet Highlighter
    "PacketHighlighter",
    "DetectionMode",
    "DetectionConfig",
    "DetectedPacket",
    "LivePacketDisplay",
]

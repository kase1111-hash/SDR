"""
UI module - User interface components.

Provides visualization components for SDR applications:
- Waterfall display with packet highlighting
- Protocol color coding
- Real-time signal detection overlay
"""

from .constellation import (
    ConstellationDisplay,
    ConstellationPoint,
    ConstellationResult,
    ConstellationStats,
    ModulationOverlay,
)
from .packet_highlighter import (
    DetectedPacket,
    DetectionConfig,
    DetectionMode,
    LivePacketDisplay,
    PacketHighlighter,
)
from .signal_meter import (
    MeterConfig,
    MeterMode,
    MeterReading,
    PowerUnit,
    SignalStrengthMeter,
)
from .time_domain import (
    DisplayMode,
    TimeDomainDisplay,
    TimeDomainResult,
)
from .waterfall import (
    PROTOCOL_COLORS,
    ColorMap,
    PacketHighlight,
    ProtocolColorScheme,
    WaterfallDisplay,
    get_protocol_color,
    list_protocol_colors,
    register_protocol_color,
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
    # Time Domain
    "TimeDomainDisplay",
    "TimeDomainResult",
    "DisplayMode",
    # Constellation
    "ConstellationDisplay",
    "ConstellationResult",
    "ConstellationStats",
    "ConstellationPoint",
    "ModulationOverlay",
    # Signal Meter
    "SignalStrengthMeter",
    "MeterReading",
    "MeterConfig",
    "PowerUnit",
    "MeterMode",
]

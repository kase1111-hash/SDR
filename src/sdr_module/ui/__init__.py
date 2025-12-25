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

from .time_domain import (
    TimeDomainDisplay,
    TimeDomainResult,
    DisplayMode,
)

from .constellation import (
    ConstellationDisplay,
    ConstellationResult,
    ConstellationStats,
    ConstellationPoint,
    ModulationOverlay,
)

from .signal_meter import (
    SignalStrengthMeter,
    MeterReading,
    MeterConfig,
    PowerUnit,
    MeterMode,
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

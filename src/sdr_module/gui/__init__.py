"""
SDR Module GUI - PyQt6-based graphical user interface.

Provides a complete GUI application for the SDR module including:
- Spectrum analyzer and waterfall display
- Device control panel
- Protocol decoder output
- Recording controls
- Frequency scanner

Optional ham radio panels (from sdr_module.ham.gui):
- HAM radio callsign identification
- SSTV image receiver (for ISS images)
- HAM-style signal meter (S-units / RST)
- QRP (low power) operations panel
- AM/FM Radio Tuner (vintage car radio style)

Usage:
    from sdr_module.gui import SDRApplication
    app = SDRApplication()
    app.run()

Or from command line:
    python -m sdr_module.gui
"""

from .app import SDRApplication
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel
from .device_dialog import DeviceDialog
from .main_window import SDRMainWindow
from .spectrum_widget import SpectrumWidget
from .waterfall_widget import WaterfallWidget

# Re-export ham radio GUI panels for backwards compatibility
try:
    from ..ham.gui import (
        AmplifierCalculator,
        AnalogMeterWidget,
        CallsignPanel,
        CompactSignalMeter,
        ImageDisplayWidget,
        PowerDisplayWidget,
        QRPPanel,
        RadioBand,
        RadioPreset,
        RadioTunerWidget,
        SignalMeterPanel,
        SSTVPanel,
        show_radio_tuner,
    )

    HAS_HAM_RADIO = True
except ImportError:
    HAS_HAM_RADIO = False

__all__ = [
    "SDRApplication",
    "SDRMainWindow",
    "SpectrumWidget",
    "WaterfallWidget",
    "ControlPanel",
    "DecoderPanel",
    "DeviceDialog",
]

# Add ham radio exports if available
if HAS_HAM_RADIO:
    __all__ += [
        "CallsignPanel",
        "SSTVPanel",
        "ImageDisplayWidget",
        "AnalogMeterWidget",
        "SignalMeterPanel",
        "CompactSignalMeter",
        "QRPPanel",
        "PowerDisplayWidget",
        "AmplifierCalculator",
        "RadioTunerWidget",
        "RadioBand",
        "RadioPreset",
        "show_radio_tuner",
    ]

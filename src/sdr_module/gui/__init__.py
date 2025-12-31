"""
SDR Module GUI - PyQt6-based graphical user interface.

Provides a complete GUI application for the SDR module including:
- Spectrum analyzer and waterfall display
- Device control panel
- Protocol decoder output
- Recording controls
- Frequency scanner
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
from .callsign_panel import CallsignPanel
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel
from .device_dialog import DeviceDialog
from .main_window import SDRMainWindow
from .qrp_panel import AmplifierCalculator, PowerDisplayWidget, QRPPanel
from .radio_tuner import RadioBand, RadioPreset, RadioTunerWidget, show_radio_tuner
from .signal_meter_widget import (
    AnalogMeterWidget,
    CompactSignalMeter,
    SignalMeterPanel,
)
from .spectrum_widget import SpectrumWidget
from .sstv_panel import ImageDisplayWidget, SSTVPanel
from .waterfall_widget import WaterfallWidget

__all__ = [
    "SDRApplication",
    "SDRMainWindow",
    "SpectrumWidget",
    "WaterfallWidget",
    "ControlPanel",
    "DecoderPanel",
    "DeviceDialog",
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

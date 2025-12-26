"""
SDR Module GUI - PyQt6-based graphical user interface.

Provides a complete GUI application for the SDR module including:
- Spectrum analyzer and waterfall display
- Device control panel
- Protocol decoder output
- Recording controls
- Frequency scanner
- HAM radio callsign identification

Usage:
    from sdr_module.gui import SDRApplication
    app = SDRApplication()
    app.run()

Or from command line:
    python -m sdr_module.gui
"""

from .main_window import SDRMainWindow
from .spectrum_widget import SpectrumWidget
from .waterfall_widget import WaterfallWidget
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel
from .device_dialog import DeviceDialog
from .callsign_panel import CallsignPanel
from .app import SDRApplication

__all__ = [
    "SDRApplication",
    "SDRMainWindow",
    "SpectrumWidget",
    "WaterfallWidget",
    "ControlPanel",
    "DecoderPanel",
    "DeviceDialog",
    "CallsignPanel",
]

"""
Frequency scanner presets and helpers.

The scanner used to be a modal dialog here, with a worker thread reading
the device while the window's display loop stood still. It is now the
non-modal Scanner tab (:mod:`sdr_module.gui.scanner_panel`), driven by the
display loop (:mod:`sdr_module.gui.scan_engine`), with the measurements in
:mod:`sdr_module.dsp.sweep`. This module keeps what the tab shares: the
band presets (with how each band is listened to), formatting and the
results table's numeric items.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTableWidgetItem

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..dsp import sweep


class ScanPreset(NamedTuple):
    """A band the scanner offers: its range, channel step and how a signal
    found there is listened to (so a hit is heard correctly: airband in
    AM, broadcast FM wide, voice FM narrow)."""

    label: str
    start_mhz: float
    end_mhz: float
    step_khz: float
    mode: str  # Demodulation > Mode item
    fm_deviation: str  # FM deviation choice ("" if not FM)
    bandwidth: str  # Bandwidth choice
    #: Signals narrower than this are not listed (the broadcast band lists
    #: stations, not the stray carriers between them).
    min_width_khz: float = 0.0

    @property
    def listening(self) -> Tuple[str, str, str]:
        return self.mode, self.fm_deviation, self.bandwidth


# Scan presets, named and ordered like the band presets of the welcome
# screen and Radio > Band Presets. The first one is the default range.
SCAN_PRESETS: Tuple[ScanPreset, ...] = (
    ScanPreset(
        "FM Broadcast (88–108 MHz)", 88.0, 108.0, 200.0, "FM", "75 kHz", "200 kHz", 30.0
    ),
    ScanPreset("NOAA Weather (162 MHz)", 162.4, 162.55, 25.0, "FM", "5 kHz", "25 kHz"),
    ScanPreset("2m Ham (144–148 MHz)", 144.0, 148.0, 12.5, "FM", "5 kHz", "25 kHz"),
    ScanPreset("70cm Ham (420–450 MHz)", 420.0, 450.0, 25.0, "FM", "5 kHz", "25 kHz"),
    ScanPreset("Airband AM (118–137 MHz)", 118.0, 137.0, 25.0, "AM", "", "10 kHz"),
    ScanPreset(
        "ISM 433 (433.05–434.79 MHz)", 433.05, 434.79, 25.0, "None (I/Q)", "", "500 kHz"
    ),
    ScanPreset("ISM 915 (902–928 MHz)", 902.0, 928.0, 100.0, "None (I/Q)", "", "2 MHz"),
)
_CUSTOM = "Custom range"


def _format_freq(freq_hz: float) -> str:
    """Always MHz, like the range fields (``1090.000 MHz``, not GHz)."""
    return f"{freq_hz / 1e6:.3f} MHz"


def _report_frequency(
    freq_hz: float, width_hz: float, step_hz: Optional[float]
) -> float:
    """See :func:`sdr_module.dsp.sweep.report_frequency`."""
    return sweep.report_frequency(freq_hz, width_hz, step_hz)


class _NumericItem(QTableWidgetItem if HAS_PYQT6 else object):
    """Table item that sorts by a numeric value instead of its text."""

    def __init__(self, text: str, value: float, align_right: bool = True):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, float(value))
        horizontal = (
            Qt.AlignmentFlag.AlignRight if align_right else Qt.AlignmentFlag.AlignLeft
        )
        self.setTextAlignment(horizontal | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)

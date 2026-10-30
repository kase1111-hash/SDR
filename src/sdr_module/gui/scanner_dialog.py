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
from .presets import scan_bands


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


# Scan presets: the bands of the window's favourite presets (the welcome
# screen's starting bands, in the same order and under the same names; see
# gui/presets.py, the one preset table). The first one is the default range.
SCAN_PRESETS: Tuple[ScanPreset, ...] = tuple(ScanPreset(*band) for band in scan_bands())
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

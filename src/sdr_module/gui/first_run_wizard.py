"""
First-run welcome / setup dialog.

Shown on first launch. Detects hardware, offers demo mode, and records
the user's starting band preference.
"""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLabel,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


BAND_PRESETS = {
    "FM Broadcast (88–108 MHz)": 100.1e6,
    "NOAA Weather (162 MHz)": 162.55e6,
    "2m Ham (144–148 MHz)": 146.52e6,
    "Airband AM (118–137 MHz)": 125.0e6,
    "70cm Ham (420–450 MHz)": 446.0e6,
    "ADS-B (1090 MHz)": 1090e6,
    "ISM 433 MHz": 433.92e6,
    "ISM 915 MHz": 915e6,
}


class FirstRunWizard(QDialog if HAS_PYQT6 else object):
    """One-shot welcome dialog offering demo mode and a starting band."""

    def __init__(self, parent=None, hardware_found: bool = False):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self.setWindowTitle("Welcome to SDR Module")
        self.resize(440, 280)

        self._selected_freq = 100.1e6

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Welcome!\n\n"
            + (
                "SDR hardware detected — you're ready to receive."
                if hardware_found
                else "No SDR hardware detected. You can still explore using Demo Mode\n"
                "(synthetic signals). Connect an RTL-SDR or HackRF One for real RX."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self._band = QComboBox()
        self._band.addItems(list(BAND_PRESETS.keys()))
        form.addRow("Start on band:", self._band)
        layout.addLayout(form)

        hint = QLabel(
            "Tip: press <b>F1</b> any time for keyboard shortcuts.\n"
            "Settings you change are remembered for next launch."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        self._selected_freq = BAND_PRESETS[self._band.currentText()]
        self.accept()

    def selected_frequency(self) -> float:
        return self._selected_freq

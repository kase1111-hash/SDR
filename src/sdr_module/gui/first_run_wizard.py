"""
First-run welcome / setup dialog.

Shown on first launch. Says whether SDR hardware was found (and how to
install a missing driver when it wasn't), offers to connect the hardware or
start Demo Mode, and records the user's starting band preference.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import set_role


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

# Glyphs that lead the status callout, as in the app's other callouts.
_OK_GLYPH = "\u2713"  # check mark
_INFO_GLYPH = "\u24d8"  # circled i

# Quick-start tips: (keys, what they do). Keys must match the shortcuts the
# main window installs (see help_dialog.SHORTCUTS).
_TIPS = (
    ("Click", "Tune to a signal on the spectrum or waterfall."),
    ("← / →", "Tune in 10 kHz steps. Add Shift for 100 kHz, Ctrl for 1 MHz."),
    ("Space", "Start or stop the receiver."),
    ("F1", "Show all keyboard shortcuts."),
)


class FirstRunWizard(QDialog if HAS_PYQT6 else object):
    """One-shot welcome dialog offering demo mode and a starting band.

    ``hardware_found`` says an SDR is plugged in (``device_name`` names it).
    ``missing_drivers`` lists ``(device label, install command)`` for the
    hardware drivers that aren't installed, as the Connect dialog shows them;
    ``None`` looks them up. With no hardware and a missing driver, the
    welcome screen explains that an SDR that is plugged in needs its driver
    first, instead of just saying nothing was detected.
    """

    if HAS_PYQT6:
        # Emitted on Get Started when "Start Demo Mode" is ticked.
        demo_mode_requested = pyqtSignal()
        # Emitted after Get Started closes the dialog when "Connect ... and
        # start receiving" is ticked (hardware found, not yet connected).
        connect_requested = pyqtSignal()

    def __init__(
        self,
        parent=None,
        hardware_found: bool = False,
        device_name: str = "",
        missing_drivers: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self.setWindowTitle("Welcome to SDR Module")
        self.setMinimumWidth(540)

        self._selected_freq = 100.1e6
        self._hardware_found = bool(hardware_found)
        self._device_name = str(device_name or "").strip()
        # Reopened (Help > Welcome) while real hardware is already open.
        self._connected = self._hardware_found and self._parent_has_hardware()
        if missing_drivers is None:
            from .device_dialog import missing_drivers as find_missing

            missing_drivers = find_missing()
        self._missing_drivers = [tuple(entry) for entry in missing_drivers]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(12)

        # Heading
        title = QLabel("Welcome to SDR Module")
        set_role(title, "display")
        layout.addWidget(title)

        subtitle = QLabel(
            "See, listen to and decode radio signals with an RTL-SDR or " "HackRF One."
        )
        subtitle.setWordWrap(True)
        set_role(subtitle, "muted")
        layout.addWidget(subtitle)

        # Hardware status
        self._status = QLabel()
        self._status.setWordWrap(True)
        # The install commands can be selected and copied.
        self._status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        set_role(self._status, "callout", "success" if self._hardware_found else "info")
        layout.addWidget(self._status)

        # Starting band
        band_group = QGroupBox("Starting Band")
        form = QFormLayout(band_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._band = QComboBox()
        self._band.addItems(list(BAND_PRESETS.keys()))
        self._band.setToolTip("The receiver tunes here when you click Get Started")
        self._band.currentTextChanged.connect(self._update_band_hint)
        form.addRow("&Band:", self._band)
        self._band_hint = QLabel()
        set_role(self._band_hint, "hint")
        form.addRow("", self._band_hint)
        self._demo_check = QCheckBox("Start &Demo Mode (simulated signals)")
        self._demo_check.setChecked(not self._hardware_found)
        self._demo_check.setToolTip(
            "Start receiving from the built-in demo device right away, no "
            "hardware needed. Switch to real hardware later with Device > "
            "Connect..."
        )
        self._connect_check = QCheckBox(
            f"&Connect {self._device_label()} and start receiving"
        )
        self._connect_check.setToolTip(
            "Get Started opens the device (you can choose its sample rate) "
            "and starts receiving. Untick to connect later with Device > "
            "Connect..."
        )
        self._connect_check.setChecked(self._hardware_found and not self._connected)
        self._connect_check.toggled.connect(self._update_status)
        # One of the two, whichever fits what was found (neither when the
        # hardware is already connected).
        if not self._hardware_found:
            form.addRow("", self._demo_check)
        elif not self._connected:
            form.addRow("", self._connect_check)
        for check in (self._demo_check, self._connect_check):
            if check.parent() is None:
                check.setParent(self)
                check.hide()
        layout.addWidget(band_group)
        self._update_band_hint(self._band.currentText())
        self._update_status()

        # Quick tips
        tips_group = QGroupBox("Quick Tips")
        tips = QGridLayout(tips_group)
        tips.setHorizontalSpacing(12)
        tips.setVerticalSpacing(6)
        tips.setColumnStretch(1, 1)
        top_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        keycaps = []
        for row, (keys, text) in enumerate(_TIPS):
            key_label = QLabel(keys)
            set_role(key_label, "keycap")
            key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tips.addWidget(key_label, row, 0, top_left)
            keycaps.append(key_label)
            text_label = QLabel(text)
            text_label.setWordWrap(True)
            tips.addWidget(text_label, row, 1, Qt.AlignmentFlag.AlignTop)
        # Equal-width keycaps keep the descriptions in one clean column.
        width = max(k.sizeHint().width() for k in keycaps)
        for key_label in keycaps:
            key_label.setFixedWidth(width)
        layout.addWidget(tips_group)

        footer = QLabel("Your settings are remembered for next time.")
        set_role(footer, "hint")
        layout.addWidget(footer)

        layout.addSpacing(4)

        btns = QDialogButtonBox()
        self._skip_btn = btns.addButton("&Skip", QDialogButtonBox.ButtonRole.RejectRole)
        self._skip_btn.setToolTip("Close without tuning or starting Demo Mode")
        self._start_btn = QPushButton("&Get Started")
        set_role(self._start_btn, "primary")
        self._start_btn.setDefault(True)
        btns.addButton(self._start_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._start_btn.setFocus()

    def _device_label(self) -> str:
        return self._device_name or "your SDR"

    def _parent_has_hardware(self) -> bool:
        """True when the parent window already has real hardware open."""
        parent = self.parent()
        return getattr(parent, "_device", None) is not None and not getattr(
            parent, "_demo_mode", True
        )

    def _status_text(self) -> str:
        name = self._device_label()
        if self._connected:
            return f"{_OK_GLYPH} Connected to {name}."
        if self._hardware_found:
            if self._connect_check.isChecked():
                return (
                    f"{_OK_GLYPH} Found {name}. Get Started connects it and "
                    "starts receiving."
                )
            return (
                f"{_OK_GLYPH} Found {name}. Connect it any time with "
                "Device > Connect..."
            )
        if self._missing_drivers:
            lines = [
                f"{_INFO_GLYPH} No SDR found. If one is plugged in, its "
                "driver isn't installed yet:"
            ]
            lines += [
                f"\u2022 {label}: {install}" for label, install in self._missing_drivers
            ]
            lines.append("Meanwhile, Demo Mode shows simulated signals.")
            return "\n".join(lines)
        return (
            f"{_INFO_GLYPH} No SDR hardware detected. Demo Mode lets you "
            "explore with simulated signals; plug in an RTL-SDR or HackRF One "
            "any time."
        )

    def _update_status(self) -> None:
        self._status.setText(self._status_text())

    def _update_band_hint(self, name: str) -> None:
        freq = BAND_PRESETS.get(name)
        if freq is None:
            self._band_hint.setText("")
            return
        self._band_hint.setText(f"Tunes to {freq / 1e6:.3f} MHz.")

    def _on_accept(self):
        self._selected_freq = BAND_PRESETS[self._band.currentText()]
        if self.wants_demo_mode():
            self._request_demo_mode()
        self.accept()
        # After closing, so the Connect dialog doesn't open over this one.
        if self.wants_connect():
            self._request_connect()

    def _request_connect(self) -> None:
        """Ask the main window to connect the hardware and start receiving."""
        self.connect_requested.emit()
        if self.receivers(self.connect_requested) == 0:
            # Nobody listening: open the parent window's Connect dialog.
            show = getattr(self.parent(), "_show_device_dialog", None)
            if callable(show):
                show(start_after=True)

    def _request_demo_mode(self) -> None:
        """Ask the main window to start Demo Mode."""
        self.demo_mode_requested.emit()
        if self.receivers(self.demo_mode_requested) == 0:
            # Nobody listening: start it on the parent window directly.
            start = getattr(self.parent(), "_start_demo_mode", None)
            if callable(start):
                start()

    def wants_demo_mode(self) -> bool:
        """True when the user ticked "Start Demo Mode"."""
        return not self._hardware_found and self._demo_check.isChecked()

    def wants_connect(self) -> bool:
        """True when the user ticked "Connect ... and start receiving"."""
        return (
            self._hardware_found
            and not self._connected
            and self._connect_check.isChecked()
        )

    def selected_frequency(self) -> float:
        return self._selected_freq

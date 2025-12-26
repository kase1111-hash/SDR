"""
Callsign identification panel for HAM radio compliance.

Provides UI controls for:
- Callsign input
- Automatic ID settings
- ID mode selection (CW, Voice, Digital)
- Manual ID trigger
"""

from __future__ import annotations

from typing import Optional
import logging

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QComboBox, QGroupBox,
        QSpinBox, QPushButton, QCheckBox, QProgressBar
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

logger = logging.getLogger(__name__)


class CallsignPanel(QWidget if HAS_PYQT6 else object):
    """
    Callsign identification control panel.

    Allows HAM operators to configure automatic callsign identification
    to comply with FCC/regulatory requirements.
    """

    if HAS_PYQT6:
        callsign_changed = pyqtSignal(str)
        id_requested = pyqtSignal()  # Manual ID request
        settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._callsign = ""
        self._is_transmitting = False
        self._id_timer = QTimer(self)
        self._id_timer.timeout.connect(self._update_countdown)
        self._seconds_until_id = 0

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Callsign group
        callsign_group = QGroupBox("Station ID (HAM)")
        callsign_layout = QVBoxLayout(callsign_group)

        # Callsign input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Callsign:"))

        self._callsign_input = QLineEdit()
        self._callsign_input.setPlaceholderText("e.g., W1AW")
        self._callsign_input.setMaxLength(10)
        self._callsign_input.setFont(QFont("Monospace", 12, QFont.Weight.Bold))
        self._callsign_input.textChanged.connect(self._on_callsign_changed)
        input_layout.addWidget(self._callsign_input)

        callsign_layout.addLayout(input_layout)

        # Auto-ID settings
        settings_layout = QGridLayout()

        # Enable auto-ID
        self._auto_id_check = QCheckBox("Auto-ID")
        self._auto_id_check.setChecked(True)
        self._auto_id_check.setToolTip("Automatically identify at required intervals")
        self._auto_id_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self._auto_id_check, 0, 0)

        # ID at start/end
        self._id_start_check = QCheckBox("ID at Start")
        self._id_start_check.setChecked(True)
        self._id_start_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self._id_start_check, 0, 1)

        self._id_end_check = QCheckBox("ID at End")
        self._id_end_check.setChecked(True)
        self._id_end_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self._id_end_check, 0, 2)

        callsign_layout.addLayout(settings_layout)

        # ID Mode and Interval
        mode_layout = QHBoxLayout()

        mode_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["CW (Morse)", "Voice", "PSK31", "RTTY"])
        self._mode_combo.currentIndexChanged.connect(self._on_settings_changed)
        mode_layout.addWidget(self._mode_combo)

        mode_layout.addWidget(QLabel("Interval:"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 10)
        self._interval_spin.setValue(10)
        self._interval_spin.setSuffix(" min")
        self._interval_spin.setToolTip("FCC requires ID at least every 10 minutes")
        self._interval_spin.valueChanged.connect(self._on_settings_changed)
        mode_layout.addWidget(self._interval_spin)

        callsign_layout.addLayout(mode_layout)

        # CW Speed (shown when CW mode selected)
        cw_layout = QHBoxLayout()
        cw_layout.addWidget(QLabel("CW Speed:"))
        self._wpm_spin = QSpinBox()
        self._wpm_spin.setRange(5, 50)
        self._wpm_spin.setValue(20)
        self._wpm_spin.setSuffix(" WPM")
        self._wpm_spin.valueChanged.connect(self._on_settings_changed)
        cw_layout.addWidget(self._wpm_spin)

        cw_layout.addWidget(QLabel("Tone:"))
        self._tone_spin = QSpinBox()
        self._tone_spin.setRange(400, 1000)
        self._tone_spin.setValue(700)
        self._tone_spin.setSuffix(" Hz")
        self._tone_spin.valueChanged.connect(self._on_settings_changed)
        cw_layout.addWidget(self._tone_spin)

        callsign_layout.addLayout(cw_layout)

        # Status and manual ID
        status_layout = QHBoxLayout()

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self._status_label)

        status_layout.addStretch()

        self._countdown_label = QLabel("")
        self._countdown_label.setFont(QFont("Monospace", 10))
        status_layout.addWidget(self._countdown_label)

        callsign_layout.addLayout(status_layout)

        # Progress bar for next ID
        self._id_progress = QProgressBar()
        self._id_progress.setRange(0, 600)
        self._id_progress.setValue(0)
        self._id_progress.setTextVisible(False)
        self._id_progress.setMaximumHeight(8)
        self._id_progress.setVisible(False)
        callsign_layout.addWidget(self._id_progress)

        # Manual ID button
        btn_layout = QHBoxLayout()

        self._id_now_btn = QPushButton("Send ID Now")
        self._id_now_btn.clicked.connect(self._on_id_now_clicked)
        self._id_now_btn.setEnabled(False)
        btn_layout.addWidget(self._id_now_btn)

        self._test_btn = QPushButton("Test ID")
        self._test_btn.setToolTip("Preview callsign ID audio")
        self._test_btn.clicked.connect(self._on_test_clicked)
        btn_layout.addWidget(self._test_btn)

        callsign_layout.addLayout(btn_layout)

        layout.addWidget(callsign_group)

    def _on_callsign_changed(self, text: str):
        """Handle callsign input change."""
        self._callsign = text.upper().strip()
        self._callsign_input.blockSignals(True)
        self._callsign_input.setText(self._callsign)
        self._callsign_input.blockSignals(False)

        # Update UI state
        has_callsign = len(self._callsign) >= 3
        self._id_now_btn.setEnabled(has_callsign and self._is_transmitting)
        self._test_btn.setEnabled(has_callsign)

        # Validate and show status
        if has_callsign:
            if self._validate_callsign(self._callsign):
                self._status_label.setText(f"Callsign: {self._callsign}")
                self._status_label.setStyleSheet("color: green;")
            else:
                self._status_label.setText(f"Warning: {self._callsign} (unusual format)")
                self._status_label.setStyleSheet("color: orange;")
        else:
            self._status_label.setText("Enter your callsign")
            self._status_label.setStyleSheet("color: gray;")

        self.callsign_changed.emit(self._callsign)

    def _validate_callsign(self, callsign: str) -> bool:
        """Basic callsign format validation."""
        if not callsign or len(callsign) < 3:
            return False
        has_letter = any(c.isalpha() for c in callsign)
        has_number = any(c.isdigit() for c in callsign)
        valid_chars = all(c.isalnum() or c == '/' for c in callsign)
        return has_letter and has_number and valid_chars

    def _on_settings_changed(self):
        """Handle settings change."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

        # Update CW controls visibility
        is_cw = self._mode_combo.currentIndex() == 0
        self._wpm_spin.setEnabled(is_cw)
        self._tone_spin.setEnabled(is_cw)

    def _on_id_now_clicked(self):
        """Handle manual ID request."""
        if self._callsign:
            self.id_requested.emit()
            self._reset_countdown()

    def _on_test_clicked(self):
        """Handle test ID button."""
        # Generate and play test ID
        try:
            from ..dsp.callsign import generate_cw_id
            import numpy as np

            if self._callsign:
                audio = generate_cw_id(
                    self._callsign,
                    wpm=self._wpm_spin.value(),
                    frequency=self._tone_spin.value()
                )
                logger.info(f"Test ID generated: {len(audio)} samples")
                # In a full implementation, this would play the audio
                self._status_label.setText(f"Test ID: DE {self._callsign}")
                self._status_label.setStyleSheet("color: blue;")
        except Exception as e:
            logger.error(f"Error generating test ID: {e}")

    def set_transmitting(self, is_transmitting: bool):
        """Set transmission state."""
        self._is_transmitting = is_transmitting
        self._id_now_btn.setEnabled(is_transmitting and len(self._callsign) >= 3)
        self._id_progress.setVisible(is_transmitting)

        if is_transmitting:
            self._status_label.setText("TX - Auto-ID active")
            self._status_label.setStyleSheet("color: red; font-weight: bold;")
            self._reset_countdown()
            self._id_timer.start(1000)  # Update every second
        else:
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet("color: gray;")
            self._id_timer.stop()
            self._countdown_label.setText("")

    def _reset_countdown(self):
        """Reset the ID countdown."""
        self._seconds_until_id = self._interval_spin.value() * 60
        self._id_progress.setMaximum(self._seconds_until_id)
        self._id_progress.setValue(self._seconds_until_id)
        self._update_countdown_display()

    def _update_countdown(self):
        """Update the countdown timer."""
        if self._seconds_until_id > 0:
            self._seconds_until_id -= 1
            self._id_progress.setValue(self._seconds_until_id)
            self._update_countdown_display()

            if self._seconds_until_id == 0:
                # Time for ID
                self.id_requested.emit()
                self._reset_countdown()

    def _update_countdown_display(self):
        """Update the countdown label."""
        minutes = self._seconds_until_id // 60
        seconds = self._seconds_until_id % 60
        self._countdown_label.setText(f"Next ID: {minutes:02d}:{seconds:02d}")

    def get_callsign(self) -> str:
        """Get the current callsign."""
        return self._callsign

    def set_callsign(self, callsign: str):
        """Set the callsign."""
        self._callsign_input.setText(callsign)

    def get_settings(self) -> dict:
        """Get all settings as a dictionary."""
        mode_map = {0: "CW", 1: "VOICE", 2: "PSK31", 3: "RTTY"}
        return {
            "callsign": self._callsign,
            "auto_id": self._auto_id_check.isChecked(),
            "id_at_start": self._id_start_check.isChecked(),
            "id_at_end": self._id_end_check.isChecked(),
            "mode": mode_map.get(self._mode_combo.currentIndex(), "CW"),
            "interval_minutes": self._interval_spin.value(),
            "cw_wpm": self._wpm_spin.value(),
            "cw_tone": self._tone_spin.value(),
        }

    def set_settings(self, settings: dict):
        """Apply settings from a dictionary."""
        if "callsign" in settings:
            self.set_callsign(settings["callsign"])
        if "auto_id" in settings:
            self._auto_id_check.setChecked(settings["auto_id"])
        if "id_at_start" in settings:
            self._id_start_check.setChecked(settings["id_at_start"])
        if "id_at_end" in settings:
            self._id_end_check.setChecked(settings["id_at_end"])
        if "mode" in settings:
            mode_index = {"CW": 0, "VOICE": 1, "PSK31": 2, "RTTY": 3}.get(settings["mode"], 0)
            self._mode_combo.setCurrentIndex(mode_index)
        if "interval_minutes" in settings:
            self._interval_spin.setValue(settings["interval_minutes"])
        if "cw_wpm" in settings:
            self._wpm_spin.setValue(settings["cw_wpm"])
        if "cw_tone" in settings:
            self._tone_spin.setValue(settings["cw_tone"])

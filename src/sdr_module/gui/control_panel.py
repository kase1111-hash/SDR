"""
Device control panel widget.

Provides controls for:
- Frequency tuning
- Gain adjustment
- Bandwidth selection
- Demodulation mode
"""

from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QSlider, QComboBox, QGroupBox,
        QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox
    )
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QDoubleValidator
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class FrequencyInput(QWidget if HAS_PYQT6 else object):
    """Frequency input widget with unit selection."""

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._frequency_hz = 100e6

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Frequency input
        self._freq_input = QDoubleSpinBox()
        self._freq_input.setRange(0.001, 9999.999)
        self._freq_input.setDecimals(6)
        self._freq_input.setValue(100.0)
        self._freq_input.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self._freq_input)

        # Unit selector
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["Hz", "kHz", "MHz", "GHz"])
        self._unit_combo.setCurrentText("MHz")
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        layout.addWidget(self._unit_combo)

    def _get_multiplier(self) -> float:
        """Get current unit multiplier."""
        unit = self._unit_combo.currentText()
        multipliers = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        return multipliers.get(unit, 1e6)

    def _on_value_changed(self, value: float):
        """Handle value change."""
        self._frequency_hz = value * self._get_multiplier()
        self.frequency_changed.emit(self._frequency_hz)

    def _on_unit_changed(self, index: int):
        """Handle unit change."""
        # Convert current value to new unit
        new_mult = self._get_multiplier()
        new_value = self._frequency_hz / new_mult
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(new_value)
        self._freq_input.blockSignals(False)

    def set_frequency(self, freq_hz: float):
        """Set frequency in Hz."""
        self._frequency_hz = freq_hz
        new_value = freq_hz / self._get_multiplier()
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(new_value)
        self._freq_input.blockSignals(False)

    def get_frequency(self) -> float:
        """Get frequency in Hz."""
        return self._frequency_hz


class ControlPanel(QWidget if HAS_PYQT6 else object):
    """
    Device control panel.

    Provides controls for frequency, gain, bandwidth, and demodulation.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)
        gain_changed = pyqtSignal(float)
        bandwidth_changed = pyqtSignal(float)
        demod_changed = pyqtSignal(str)
        recording_started = pyqtSignal(str)  # format
        recording_stopped = pyqtSignal()

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)

        # Frequency group
        freq_group = QGroupBox("Frequency")
        freq_layout = QVBoxLayout(freq_group)

        # Main frequency
        self._freq_input = FrequencyInput()
        self._freq_input.frequency_changed.connect(self.frequency_changed)
        freq_layout.addWidget(self._freq_input)

        # Quick tune buttons
        quick_layout = QHBoxLayout()
        for offset in [-1e6, -100e3, -10e3, 10e3, 100e3, 1e6]:
            btn = QPushButton(self._format_offset(offset))
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, o=offset: self._quick_tune(o))
            quick_layout.addWidget(btn)
        freq_layout.addLayout(quick_layout)

        layout.addWidget(freq_group)

        # Gain group
        gain_group = QGroupBox("Gain")
        gain_layout = QGridLayout(gain_group)

        # RF gain
        gain_layout.addWidget(QLabel("RF:"), 0, 0)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, 50)
        self._gain_slider.setValue(20)
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_layout.addWidget(self._gain_slider, 0, 1)

        self._gain_label = QLabel("20.0 dB")
        gain_layout.addWidget(self._gain_label, 0, 2)

        # AGC checkbox
        self._agc_check = QCheckBox("AGC")
        self._agc_check.stateChanged.connect(self._on_agc_changed)
        gain_layout.addWidget(self._agc_check, 1, 0, 1, 3)

        layout.addWidget(gain_group)

        # Bandwidth group
        bw_group = QGroupBox("Bandwidth")
        bw_layout = QHBoxLayout(bw_group)

        self._bw_combo = QComboBox()
        self._bw_combo.addItems([
            "10 kHz", "25 kHz", "50 kHz", "100 kHz", "200 kHz",
            "500 kHz", "1 MHz", "2 MHz", "2.4 MHz"
        ])
        self._bw_combo.setCurrentText("200 kHz")
        self._bw_combo.currentTextChanged.connect(self._on_bandwidth_changed)
        bw_layout.addWidget(self._bw_combo)

        layout.addWidget(bw_group)

        # Demodulation group
        demod_group = QGroupBox("Demodulation")
        demod_layout = QVBoxLayout(demod_group)

        self._demod_combo = QComboBox()
        self._demod_combo.addItems([
            "None (I/Q)", "AM", "FM", "USB", "LSB", "CW"
        ])
        self._demod_combo.currentTextChanged.connect(self._on_demod_changed)
        demod_layout.addWidget(self._demod_combo)

        # FM deviation (for FM mode)
        fm_layout = QHBoxLayout()
        fm_layout.addWidget(QLabel("FM Dev:"))
        self._fm_dev_combo = QComboBox()
        self._fm_dev_combo.addItems(["5 kHz", "12.5 kHz", "25 kHz", "75 kHz"])
        self._fm_dev_combo.setCurrentText("25 kHz")
        fm_layout.addWidget(self._fm_dev_combo)
        demod_layout.addLayout(fm_layout)

        layout.addWidget(demod_group)

        # Squelch
        squelch_group = QGroupBox("Squelch")
        squelch_layout = QHBoxLayout(squelch_group)

        self._squelch_slider = QSlider(Qt.Orientation.Horizontal)
        self._squelch_slider.setRange(-120, 0)
        self._squelch_slider.setValue(-80)
        squelch_layout.addWidget(self._squelch_slider)

        self._squelch_label = QLabel("-80 dB")
        self._squelch_slider.valueChanged.connect(
            lambda v: self._squelch_label.setText(f"{v} dB")
        )
        squelch_layout.addWidget(self._squelch_label)

        layout.addWidget(squelch_group)

        # Recording group
        record_group = QGroupBox("Recording")
        record_layout = QVBoxLayout(record_group)

        # Record buttons
        btn_layout = QHBoxLayout()
        self._record_btn = QPushButton("⏺ Record")
        self._record_btn.setCheckable(True)
        self._record_btn.toggled.connect(self._on_record_toggled)
        btn_layout.addWidget(self._record_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setEnabled(False)
        btn_layout.addWidget(self._pause_btn)
        record_layout.addLayout(btn_layout)

        # Recording format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["Raw IQ (Complex64)", "WAV (16-bit)", "SigMF"])
        format_layout.addWidget(self._format_combo)
        record_layout.addLayout(format_layout)

        # Recording status
        self._record_status = QLabel("Ready")
        record_layout.addWidget(self._record_status)

        self._record_time = QLabel("00:00:00")
        record_layout.addWidget(self._record_time)

        layout.addWidget(record_group)

        # Add stretch at bottom
        layout.addStretch()

    def _format_offset(self, offset: float) -> str:
        """Format frequency offset for button label."""
        if abs(offset) >= 1e6:
            return f"{'+' if offset > 0 else ''}{offset/1e6:.0f}M"
        elif abs(offset) >= 1e3:
            return f"{'+' if offset > 0 else ''}{offset/1e3:.0f}k"
        else:
            return f"{'+' if offset > 0 else ''}{offset:.0f}"

    def _quick_tune(self, offset: float):
        """Quick tune by offset."""
        current = self._freq_input.get_frequency()
        self._freq_input.set_frequency(current + offset)
        self.frequency_changed.emit(current + offset)

    def _on_gain_changed(self, value: int):
        """Handle gain change."""
        gain = float(value)
        self._gain_label.setText(f"{gain:.1f} dB")
        self.gain_changed.emit(gain)

    def _on_agc_changed(self, state: int):
        """Handle AGC toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self._gain_slider.setEnabled(not enabled)
        if enabled:
            self.gain_changed.emit(-1)  # -1 indicates AGC

    def _on_bandwidth_changed(self, text: str):
        """Handle bandwidth change."""
        # Parse bandwidth string
        value = float(text.split()[0])
        unit = text.split()[1]
        multiplier = {"kHz": 1e3, "MHz": 1e6}.get(unit, 1)
        self.bandwidth_changed.emit(value * multiplier)

    def _on_demod_changed(self, text: str):
        """Handle demodulation mode change."""
        self.demod_changed.emit(text)

    def set_frequency(self, freq_hz: float):
        """Set frequency."""
        self._freq_input.set_frequency(freq_hz)

    def set_gain(self, gain_db: float):
        """Set gain."""
        self._gain_slider.setValue(int(gain_db))

    def _on_record_toggled(self, checked: bool):
        """Handle record button toggle."""
        if checked:
            self._record_btn.setText("⏹ Stop")
            self._pause_btn.setEnabled(True)
            self._record_status.setText("Recording...")
            self._format_combo.setEnabled(False)
            self.recording_started.emit(self._format_combo.currentText())
        else:
            self._record_btn.setText("⏺ Record")
            self._pause_btn.setEnabled(False)
            self._pause_btn.setChecked(False)
            self._record_status.setText("Ready")
            self._record_time.setText("00:00:00")
            self._format_combo.setEnabled(True)
            self.recording_stopped.emit()

    def update_record_time(self, seconds: int):
        """Update recording time display."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        self._record_time.setText(f"{hours:02d}:{minutes:02d}:{secs:02d}")

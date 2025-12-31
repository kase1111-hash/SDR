"""
QRP (Low Power) Operations Panel.

Provides controls for QRP operation:
- Power display in watts/mW/dBm
- TX power limiter
- Amplifier chain calculator
- QRP compliance indicator
- Miles-per-watt tracker
"""

from __future__ import annotations

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..dsp.qrp import QRPController, dbm_to_watts, format_power, format_power_verbose


class PowerDisplayWidget(QWidget if HAS_PYQT6 else object):
    """Widget showing power in multiple formats."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Watts display (big)
        self._watts_label = QLabel("0 mW")
        self._watts_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #4f4;"
        )
        layout.addWidget(self._watts_label, 0, 0, 1, 2)

        # dBm display
        layout.addWidget(QLabel("dBm:"), 1, 0)
        self._dbm_label = QLabel("0 dBm")
        self._dbm_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self._dbm_label, 1, 1)

        # QRP status
        layout.addWidget(QLabel("Status:"), 2, 0)
        self._status_label = QLabel("QRPp")
        self._status_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ff4;"
        )
        layout.addWidget(self._status_label, 2, 1)

    def set_power(self, dbm: float, mode: str = "CW") -> None:
        """Update power display."""
        watts = dbm_to_watts(dbm)

        # Format watts
        self._watts_label.setText(format_power(watts))
        self._dbm_label.setText(f"{dbm:+.1f} dBm")

        # Determine QRP status and color
        if watts <= 0.001:
            status = "QRPp (mW)"
            color = "#4f4"  # Green
        elif watts <= 5.0:
            status = "QRP" if mode.upper() in ("CW", "RTTY", "FT8") else "QRP"
            color = "#4f4"  # Green
        elif watts <= 10.0:
            if mode.upper() in ("SSB", "FM", "AM"):
                status = "QRP (SSB)"
                color = "#4f4"  # Green
            else:
                status = "Low Power"
                color = "#ff4"  # Yellow
        elif watts <= 100.0:
            status = "Low Power"
            color = "#ff4"  # Yellow
        else:
            status = "QRO"
            color = "#f44"  # Red

        self._status_label.setText(status)
        self._status_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color};"
        )

        # Update watts color based on status
        self._watts_label.setStyleSheet(
            f"font-size: 28px; font-weight: bold; color: {color};"
        )


class AmplifierCalculator(QWidget if HAS_PYQT6 else object):
    """Amplifier chain power calculator."""

    if HAS_PYQT6:
        power_changed = pyqtSignal(float)  # Output power in dBm

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._setup_ui()
        self._calculate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Input power (SDR output)
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("SDR Output:"))
        self._input_spin = QSpinBox()
        self._input_spin.setRange(-20, 20)
        self._input_spin.setValue(0)
        self._input_spin.setSuffix(" dBm")
        self._input_spin.valueChanged.connect(self._calculate)
        input_layout.addWidget(self._input_spin)

        self._input_watts = QLabel("(1 mW)")
        self._input_watts.setStyleSheet("color: #888;")
        input_layout.addWidget(self._input_watts)
        layout.addLayout(input_layout)

        # Driver stage
        driver_layout = QHBoxLayout()
        self._driver_check = QCheckBox("Driver:")
        self._driver_check.setChecked(True)
        self._driver_check.stateChanged.connect(self._calculate)
        driver_layout.addWidget(self._driver_check)

        self._driver_spin = QSpinBox()
        self._driver_spin.setRange(0, 30)
        self._driver_spin.setValue(20)
        self._driver_spin.setSuffix(" dB")
        self._driver_spin.valueChanged.connect(self._calculate)
        driver_layout.addWidget(self._driver_spin)

        self._driver_out = QLabel("→ 100 mW")
        self._driver_out.setStyleSheet("color: #888;")
        driver_layout.addWidget(self._driver_out)
        layout.addLayout(driver_layout)

        # PA stage
        pa_layout = QHBoxLayout()
        self._pa_check = QCheckBox("PA:")
        self._pa_check.setChecked(True)
        self._pa_check.stateChanged.connect(self._calculate)
        pa_layout.addWidget(self._pa_check)

        self._pa_spin = QSpinBox()
        self._pa_spin.setRange(0, 30)
        self._pa_spin.setValue(10)
        self._pa_spin.setSuffix(" dB")
        self._pa_spin.valueChanged.connect(self._calculate)
        pa_layout.addWidget(self._pa_spin)

        self._pa_out = QLabel("→ 1 W")
        self._pa_out.setStyleSheet("color: #888;")
        pa_layout.addWidget(self._pa_out)
        layout.addLayout(pa_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

        # Output
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self._output_label = QLabel("1 W (+30 dBm)")
        self._output_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #4f4;"
        )
        output_layout.addWidget(self._output_label)
        layout.addLayout(output_layout)

        # DC Power estimate
        dc_layout = QHBoxLayout()
        dc_layout.addWidget(QLabel("Est. DC Power:"))
        self._dc_label = QLabel("2 W")
        self._dc_label.setStyleSheet("color: #888;")
        dc_layout.addWidget(self._dc_label)
        layout.addLayout(dc_layout)

    def _calculate(self):
        """Recalculate power chain."""
        input_dbm = self._input_spin.value()
        self._input_watts.setText(f"({format_power(dbm_to_watts(input_dbm))})")

        current_dbm = input_dbm
        dc_power = 0.0

        # Driver stage
        if self._driver_check.isChecked():
            gain = self._driver_spin.value()
            current_dbm += gain
            watts = dbm_to_watts(current_dbm)
            dc_power += watts / 0.5  # 50% efficiency
            self._driver_out.setText(f"→ {format_power(watts)}")
            self._driver_out.setStyleSheet("color: #888;")
            self._driver_spin.setEnabled(True)
        else:
            self._driver_out.setText("(bypassed)")
            self._driver_out.setStyleSheet("color: #666;")
            self._driver_spin.setEnabled(False)

        # PA stage
        if self._pa_check.isChecked():
            gain = self._pa_spin.value()
            current_dbm += gain
            watts = dbm_to_watts(current_dbm)
            dc_power += watts / 0.5  # 50% efficiency
            self._pa_out.setText(f"→ {format_power(watts)}")
            self._pa_out.setStyleSheet("color: #888;")
            self._pa_spin.setEnabled(True)
        else:
            self._pa_out.setText("(bypassed)")
            self._pa_out.setStyleSheet("color: #666;")
            self._pa_spin.setEnabled(False)

        # Output
        output_watts = dbm_to_watts(current_dbm)
        self._output_label.setText(format_power_verbose(output_watts, current_dbm))

        # Color based on QRP status
        if output_watts <= 5.0:
            color = "#4f4"  # Green - QRP
        elif output_watts <= 100.0:
            color = "#ff4"  # Yellow - Low power
        else:
            color = "#f44"  # Red - QRO

        self._output_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {color};"
        )

        # DC power
        self._dc_label.setText(f"{dc_power:.1f} W")

        # Emit signal
        self.power_changed.emit(current_dbm)

    def get_output_dbm(self) -> float:
        """Get calculated output power in dBm."""
        result = self._input_spin.value()
        if self._driver_check.isChecked():
            result += self._driver_spin.value()
        if self._pa_check.isChecked():
            result += self._pa_spin.value()
        return result


class QRPPanel(QWidget if HAS_PYQT6 else object):
    """
    Complete QRP operations panel.

    Provides:
    - Power display in multiple formats
    - TX power limiter
    - Amplifier calculator
    - QRP compliance status
    - Miles-per-watt tracker
    """

    if HAS_PYQT6:
        power_limit_changed = pyqtSignal(float)  # New limit in watts

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._qrp = QRPController()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Power display group
        power_group = QGroupBox("TX Power")
        power_layout = QVBoxLayout(power_group)

        self._power_display = PowerDisplayWidget()
        power_layout.addWidget(self._power_display)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["CW", "SSB", "FM", "FT8", "RTTY", "AM"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        mode_layout.addStretch()
        power_layout.addLayout(mode_layout)

        layout.addWidget(power_group)

        # Power limiter group
        limit_group = QGroupBox("TX Power Limit")
        limit_layout = QVBoxLayout(limit_group)

        self._limit_check = QCheckBox("Enable power limit")
        self._limit_check.stateChanged.connect(self._on_limit_toggled)
        limit_layout.addWidget(self._limit_check)

        limit_val_layout = QHBoxLayout()
        limit_val_layout.addWidget(QLabel("Max power:"))

        self._limit_spin = QDoubleSpinBox()
        self._limit_spin.setRange(0.001, 1500.0)
        self._limit_spin.setValue(5.0)
        self._limit_spin.setSuffix(" W")
        self._limit_spin.setDecimals(3)
        self._limit_spin.setEnabled(False)
        self._limit_spin.valueChanged.connect(self._on_limit_changed)
        limit_val_layout.addWidget(self._limit_spin)

        # Quick limit buttons
        for watts in [1.0, 5.0, 10.0]:
            btn = QPushButton(f"{watts:.0f}W")
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, w=watts: self._set_quick_limit(w))
            limit_val_layout.addWidget(btn)

        limit_layout.addLayout(limit_val_layout)

        # QRP presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        qrpp_btn = QPushButton("QRPp (1W)")
        qrpp_btn.clicked.connect(lambda: self._set_quick_limit(1.0))
        preset_layout.addWidget(qrpp_btn)

        qrp_cw_btn = QPushButton("QRP CW (5W)")
        qrp_cw_btn.clicked.connect(lambda: self._set_quick_limit(5.0))
        preset_layout.addWidget(qrp_cw_btn)

        qrp_ssb_btn = QPushButton("QRP SSB (10W)")
        qrp_ssb_btn.clicked.connect(lambda: self._set_quick_limit(10.0))
        preset_layout.addWidget(qrp_ssb_btn)

        limit_layout.addLayout(preset_layout)

        layout.addWidget(limit_group)

        # Amplifier calculator
        amp_group = QGroupBox("Amplifier Chain Calculator")
        amp_layout = QVBoxLayout(amp_group)

        self._amp_calc = AmplifierCalculator()
        self._amp_calc.power_changed.connect(self._on_calc_power_changed)
        amp_layout.addWidget(self._amp_calc)

        layout.addWidget(amp_group)

        # Miles per watt tracker
        mpw_group = QGroupBox("Miles Per Watt")
        mpw_layout = QGridLayout(mpw_group)

        mpw_layout.addWidget(QLabel("Distance:"), 0, 0)
        self._distance_spin = QSpinBox()
        self._distance_spin.setRange(1, 20000)
        self._distance_spin.setValue(500)
        self._distance_spin.setSuffix(" mi")
        mpw_layout.addWidget(self._distance_spin, 0, 1)

        mpw_layout.addWidget(QLabel("Power:"), 0, 2)
        self._mpw_power_spin = QDoubleSpinBox()
        self._mpw_power_spin.setRange(0.001, 100)
        self._mpw_power_spin.setValue(5.0)
        self._mpw_power_spin.setSuffix(" W")
        mpw_layout.addWidget(self._mpw_power_spin, 0, 3)

        self._log_qso_btn = QPushButton("Log QSO")
        self._log_qso_btn.clicked.connect(self._on_log_qso)
        mpw_layout.addWidget(self._log_qso_btn, 0, 4)

        # MPW display
        mpw_layout.addWidget(QLabel("This QSO:"), 1, 0)
        self._mpw_label = QLabel("100 MPW")
        self._mpw_label.setStyleSheet("font-weight: bold;")
        mpw_layout.addWidget(self._mpw_label, 1, 1)

        mpw_layout.addWidget(QLabel("Best:"), 1, 2)
        self._best_mpw_label = QLabel("0 MPW")
        self._best_mpw_label.setStyleSheet("color: #4f4; font-weight: bold;")
        mpw_layout.addWidget(self._best_mpw_label, 1, 3)

        mpw_layout.addWidget(QLabel("QSOs:"), 1, 4)
        self._qso_count_label = QLabel("0")
        mpw_layout.addWidget(self._qso_count_label, 1, 5)

        layout.addWidget(mpw_group)

        # Stretch at bottom
        layout.addStretch()

        # Initial calculation
        self._update_mpw_display()

    def _on_mode_changed(self, mode: str):
        """Handle mode change."""
        # Update power display with new mode
        dbm = self._amp_calc.get_output_dbm()
        self._power_display.set_power(dbm, mode)

    def _on_limit_toggled(self, state: int):
        """Handle limit checkbox toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self._limit_spin.setEnabled(enabled)

        if enabled:
            self._qrp.set_power_limit(self._limit_spin.value())
        else:
            self._qrp.disable_power_limit()

    def _on_limit_changed(self, value: float):
        """Handle limit value change."""
        if self._limit_check.isChecked():
            self._qrp.set_power_limit(value)
            self.power_limit_changed.emit(value)

    def _set_quick_limit(self, watts: float):
        """Set a quick power limit."""
        self._limit_check.setChecked(True)
        self._limit_spin.setValue(watts)
        self._qrp.set_power_limit(watts)

    def _on_calc_power_changed(self, dbm: float):
        """Handle calculator power change."""
        mode = self._mode_combo.currentText()
        self._power_display.set_power(dbm, mode)

    def _on_log_qso(self):
        """Log a QSO for MPW tracking."""
        distance = self._distance_spin.value()
        power = self._mpw_power_spin.value()

        self._qrp.log_qso(distance, power)
        self._update_mpw_display()

    def _update_mpw_display(self):
        """Update miles-per-watt display."""
        distance = self._distance_spin.value()
        power = self._mpw_power_spin.value()
        current_mpw = distance / power if power > 0 else 0

        self._mpw_label.setText(f"{current_mpw:.0f} MPW")

        stats = self._qrp.get_statistics()
        self._best_mpw_label.setText(f"{stats['best_mpw']:.0f} MPW")
        self._qso_count_label.setText(str(stats["total_qsos"]))

    def get_controller(self) -> QRPController:
        """Get the QRP controller instance."""
        return self._qrp

    def set_power(self, dbm: float):
        """Set current power for display."""
        mode = self._mode_combo.currentText()
        self._power_display.set_power(dbm, mode)


__all__ = [
    "PowerDisplayWidget",
    "AmplifierCalculator",
    "QRPPanel",
]

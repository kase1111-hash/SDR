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

import math
from typing import Any, Optional, Tuple

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..qrp import QRP_LIMITS, QRPController, dbm_to_watts

# Modes held to the stricter CW/digital QRP limit (matches QRPController).
_CW_DIGITAL_MODES = ("CW", "RTTY", "PSK", "FT8", "FT4", "JT65", "WSPR")
# Power within this many dB of a limit counts as "at" the limit. Whole-dB
# steps land just beside round watt values (+37 dBm is 5.01 W), which would
# otherwise show "5.0 W" next to "Above the 5 W QRP limit".
_LIMIT_TOLERANCE_DB = 0.05


_LIMIT_TIP = "Maximum transmit power you intend to use"
_LIMIT_OFF_TIP = (
    "Turn on \u201cWarn when output exceeds the limit\u201d (or pick a preset) "
    "to set a maximum power."
)


def _theme() -> Any:
    """The GUI theme helpers, imported lazily.

    ``sdr_module.gui`` imports this panel while it initialises, so importing
    ``sdr_module.gui.themes`` at module level here would be circular.
    """
    from ...gui import themes

    return themes


def _trim(number: str) -> str:
    """Drop a fractional part that is all zeros ("5.0" -> "5", "2.5" stays)."""
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


# (smallest value in watts, watts per unit, unit). The thresholds sit just
# under 1 so a value that would round to "1000" of a unit uses the next unit
# up: 0.9996 W reads "1 W", not "1000 mW".
_POWER_UNITS = (
    (0.9995, 1.0, "W"),
    (0.9995e-3, 1e-3, "mW"),
    (0.9995e-6, 1e-6, "µW"),
    (0.9995e-9, 1e-9, "nW"),
)


def format_watts(watts: float) -> str:
    """Power in the panel's one watt format.

    One decimal below 100 of a unit, none above, trailing ".0" dropped:
    ``"5 W"``, ``"1.3 W"``, ``"12.6 W"``, ``"250 mW"``, ``"1 mW"``.
    """
    watts = float(watts)
    if watts <= 0.0 or not math.isfinite(watts):
        return "0 W"
    scale, unit = 1e-12, "pW"
    for threshold, unit_scale, unit_name in _POWER_UNITS:
        if watts >= threshold:
            scale, unit = unit_scale, unit_name
            break
    value = watts / scale
    number = f"{value:.0f}" if value >= 99.95 else _trim(f"{value:.1f}")
    return f"{number} {unit}"


def format_dbm(dbm: float) -> str:
    """Power in dBm, signed, with one decimal only when it is not zero
    (``"+30 dBm"``, ``"+30.5 dBm"``, ``"0 dBm"``)."""
    number = _trim(f"{float(dbm):+.1f}")
    if number in ("+0", "-0"):
        number = "0"
    return f"{number} dBm"


def qrp_limit_for_mode(mode: str) -> float:
    """QRP ceiling in watts for ``mode`` (5 W CW/digital, 10 W phone)."""
    if mode.upper() in _CW_DIGITAL_MODES:
        return QRP_LIMITS.qrp_cw_watts
    return QRP_LIMITS.qrp_ssb_watts


def within_limit(watts: float, limit: float) -> bool:
    """True when ``watts`` is at or below ``limit`` (to within 0.05 dB)."""
    return watts <= limit * 10.0 ** (_LIMIT_TOLERANCE_DB / 10.0)


def classify_qrp(watts: float, mode: str = "CW") -> Tuple[str, str, str]:
    """Classify ``watts`` for ``mode``.

    Returns:
        ``(label, tone, explanation)``; ``tone`` is a theme tone name.
    """
    mode = (mode or "CW").upper()
    limit = qrp_limit_for_mode(mode)
    if within_limit(watts, QRP_LIMITS.qrpp_watts):
        return "QRPp", "success", f"QRPp: {format_watts(QRP_LIMITS.qrpp_watts)} or less"
    if within_limit(watts, limit):
        return (
            "QRP",
            "success",
            f"Within the {format_watts(limit)} QRP limit for {mode}",
        )
    if within_limit(watts, QRP_LIMITS.low_power_watts):
        return (
            "Low Power",
            "warning",
            f"Above the {format_watts(limit)} QRP limit for {mode}",
        )
    return (
        "QRO",
        "danger",
        f"Above {format_watts(QRP_LIMITS.low_power_watts)}: high power (QRO)",
    )


class _WattSpinBox(QDoubleSpinBox if HAS_PYQT6 else object):
    """Watt spin box that shows ``5 W`` / ``0.25 W`` rather than ``5.000 W``.

    It keeps three decimals (1 mW steps) for input but drops trailing zeros
    from the displayed value, like the panel's other watt readouts.
    """

    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        point = self.locale().decimalPoint()
        if point and point in text:
            text = text.rstrip("0").rstrip(point)
        return text


def _match_label_heights(form: "QFormLayout") -> None:
    """Make each form label as tall as its field.

    QFormLayout caps a label at 7/4 of its own height, which leaves it a few
    pixels above the centre of a taller spin box.
    """
    for row in range(form.rowCount()):
        label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
        if label_item is None or field_item is None:
            continue
        label = label_item.widget()
        if label is None:
            continue
        field = field_item.widget()
        if field is not None:
            field.ensurePolished()
        label.setMinimumHeight(field_item.sizeHint().height())


def _make_form(parent: Optional["QWidget"] = None) -> "QFormLayout":
    form = QFormLayout(parent) if parent is not None else QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(8)
    return form


def _readout(text: str = "", role: str = "muted") -> "QLabel":
    """Right-aligned result label for a calculator row."""
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    _theme().set_role(label, role)
    return label


class PowerDisplayWidget(QWidget if HAS_PYQT6 else object):
    """Widget showing power in multiple formats."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        t = _theme()

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        # Watts display (big)
        self._watts_label = QLabel("0 mW")
        t.set_role(self._watts_label, "display")
        self._watts_label.setToolTip(
            "Transmit power at the antenna, as worked out by the Amplifier "
            "Chain below"
        )
        layout.addWidget(self._watts_label, 0, 0)

        # QRP class badge
        self._status_label = QLabel("QRPp")
        t.set_role(self._status_label, "badge", "success")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(
            self._status_label,
            0,
            1,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        # dBm display
        self._dbm_label = QLabel("0 dBm")
        t.set_role(self._dbm_label, "muted")
        self._dbm_label.setToolTip("Transmit power in dBm (0 dBm = 1 mW)")
        layout.addWidget(self._dbm_label, 1, 0, 1, 2)

        # Why the badge says what it says.
        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        t.set_role(self._detail_label, "hint")
        layout.addWidget(self._detail_label, 2, 0, 1, 2)
        layout.setColumnStretch(0, 1)

        self.set_power(0.0)

    def set_power(self, dbm: float, mode: str = "CW") -> None:
        """Update power display."""
        watts = dbm_to_watts(dbm)
        label, tone, detail = classify_qrp(watts, mode)

        self._watts_label.setText(format_watts(watts))
        self._dbm_label.setText(format_dbm(dbm))
        self._status_label.setText(label)
        self._detail_label.setText(detail)
        self._status_label.setToolTip(detail)

        t = _theme()
        t.set_tone(self._status_label, tone)
        t.set_tone(self._watts_label, tone)


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
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.setColumnStretch(1, 1)

        # Input power (SDR output)
        sdr_label = QLabel("SDR output:")
        self._label_column = [sdr_label]
        layout.addWidget(sdr_label, 0, 0)
        self._input_spin = QSpinBox()
        self._input_spin.setRange(-20, 20)
        self._input_spin.setValue(0)
        self._input_spin.setSuffix(" dBm")
        self._input_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._input_spin.setToolTip(
            "RF output of the SDR (exciter). A HackRF One gives roughly "
            "0 to +15 dBm depending on frequency."
        )
        self._input_spin.valueChanged.connect(self._calculate)
        sdr_label.setBuddy(self._input_spin)
        layout.addWidget(self._input_spin, 0, 1)
        self._input_watts = _readout("1 mW")
        layout.addWidget(self._input_watts, 0, 2)

        # Driver stage
        self._driver_check = QCheckBox("Driver:")
        self._driver_check.setChecked(True)
        self._driver_check.setToolTip("Include a driver amplifier stage")
        self._driver_check.stateChanged.connect(self._calculate)
        layout.addWidget(self._driver_check, 1, 0)

        self._driver_spin = QSpinBox()
        self._driver_spin.setRange(0, 30)
        self._driver_spin.setValue(20)
        self._driver_spin.setSuffix(" dB")
        self._driver_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._driver_spin.setToolTip("Gain of the driver stage")
        # Labelled by a checkbox, which cannot be a buddy: name it directly.
        self._driver_spin.setAccessibleName("Driver gain")
        self._driver_spin.valueChanged.connect(self._calculate)
        layout.addWidget(self._driver_spin, 1, 1)

        self._driver_out = _readout("100 mW")
        layout.addWidget(self._driver_out, 1, 2)

        # PA stage
        self._pa_check = QCheckBox("PA:")
        self._pa_check.setChecked(True)
        self._pa_check.setToolTip("Include the final power amplifier (PA)")
        self._pa_check.stateChanged.connect(self._calculate)
        layout.addWidget(self._pa_check, 2, 0)

        self._pa_spin = QSpinBox()
        self._pa_spin.setRange(0, 30)
        self._pa_spin.setValue(10)
        self._pa_spin.setSuffix(" dB")
        self._pa_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._pa_spin.setToolTip("Gain of the power amplifier stage")
        self._pa_spin.setAccessibleName("PA gain")
        self._pa_spin.valueChanged.connect(self._calculate)
        layout.addWidget(self._pa_spin, 2, 1)

        self._pa_out = _readout("1 W")
        layout.addWidget(self._pa_out, 2, 2)

        # Result column must fit "(bypassed)" and "79.4 mW" without jitter.
        for label in (self._input_watts, self._driver_out, self._pa_out):
            label.ensurePolished()
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance("(bypassed)"))

        # A little air between the stages and the totals.
        layout.setRowMinimumHeight(3, 2)

        # Output
        output_caption = QLabel("Output:")
        layout.addWidget(output_caption, 4, 0)
        self._output_label = _readout("1 W (+30 dBm)", "value")
        self._output_label.setToolTip("Power at the antenna connector")
        layout.addWidget(self._output_label, 4, 1, 1, 2)

        # DC Power estimate
        dc_caption = QLabel("Est. DC input:")
        dc_caption.setToolTip(
            "Rough supply power needed, assuming each stage is 50 % efficient"
        )
        layout.addWidget(dc_caption, 5, 0)
        self._label_column += [
            self._driver_check,
            self._pa_check,
            output_caption,
            dc_caption,
        ]
        self._dc_label = _readout("2.2 W")
        self._dc_label.setToolTip(dc_caption.toolTip())
        layout.addWidget(self._dc_label, 5, 1, 1, 2)

    def label_column_width(self) -> int:
        """Width the left (label / stage) column needs."""
        for widget in self._label_column:
            widget.ensurePolished()
        return max(w.sizeHint().width() for w in self._label_column)

    def set_label_column_width(self, width: int) -> None:
        """Widen the left column so fields line up with other forms."""
        layout = self.layout()
        if isinstance(layout, QGridLayout):
            layout.setColumnMinimumWidth(0, width)

    def _calculate(self):
        """Recalculate power chain."""
        t = _theme()
        input_dbm = self._input_spin.value()
        self._input_watts.setText(format_watts(dbm_to_watts(input_dbm)))

        current_dbm = input_dbm
        dc_power = 0.0

        # Driver stage
        if self._driver_check.isChecked():
            gain = self._driver_spin.value()
            current_dbm += gain
            watts = dbm_to_watts(current_dbm)
            dc_power += watts / 0.5  # 50% efficiency
            self._driver_out.setText(format_watts(watts))
            t.set_tone(self._driver_out, None)
            self._driver_spin.setEnabled(True)
        else:
            self._driver_out.setText("(bypassed)")
            t.set_tone(self._driver_out, "muted")
            self._driver_spin.setEnabled(False)

        # PA stage
        if self._pa_check.isChecked():
            gain = self._pa_spin.value()
            current_dbm += gain
            watts = dbm_to_watts(current_dbm)
            dc_power += watts / 0.5  # 50% efficiency
            self._pa_out.setText(format_watts(watts))
            t.set_tone(self._pa_out, None)
            self._pa_spin.setEnabled(True)
        else:
            self._pa_out.setText("(bypassed)")
            t.set_tone(self._pa_out, "muted")
            self._pa_spin.setEnabled(False)

        # Output
        output_watts = dbm_to_watts(current_dbm)
        self._output_label.setText(
            f"{format_watts(output_watts)} ({format_dbm(current_dbm)})"
        )

        # DC power
        self._dc_label.setText(format_watts(dc_power))

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
        self._current_dbm = 0.0
        self._setup_ui()

    def _setup_ui(self):
        t = _theme()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # All content scrolls, so the groups keep their natural height in a
        # short tab instead of being crushed on top of each other.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Not a Tab stop of its own: focus goes straight to the controls.
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        outer.addWidget(self._scroll)

        content = QWidget()
        self._scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- TX power -----------------------------------------------------
        power_group = QGroupBox("TX Power")
        power_layout = QVBoxLayout(power_group)
        power_layout.setSpacing(8)

        self._power_display = PowerDisplayWidget()
        power_layout.addWidget(self._power_display)

        self._limit_status = QLabel("")
        self._limit_status.setWordWrap(True)
        t.set_role(self._limit_status, "callout", "success")
        self._limit_status.setVisible(False)
        power_layout.addWidget(self._limit_status)

        mode_form = _make_form()
        self._forms = [mode_form]
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["CW", "SSB", "FM", "FT8", "RTTY", "AM"])
        self._mode_combo.setToolTip(
            "Operating mode. QRP means 5 W or less for CW and digital modes, "
            "10 W or less for phone (SSB, FM, AM)."
        )
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._mode_label = QLabel("Mode:")
        self._mode_label.setBuddy(self._mode_combo)
        mode_form.addRow(self._mode_label, self._mode_combo)
        power_layout.addLayout(mode_form)

        layout.addWidget(power_group)

        # --- Amplifier calculator ----------------------------------------
        amp_group = QGroupBox("Amplifier Chain")
        amp_group.setToolTip("Work out the output power of an SDR + amplifier chain")
        amp_layout = QVBoxLayout(amp_group)

        self._amp_calc = AmplifierCalculator()
        self._amp_calc.power_changed.connect(self._on_calc_power_changed)
        amp_layout.addWidget(self._amp_calc)

        layout.addWidget(amp_group)

        # --- Power limit --------------------------------------------------
        limit_group = QGroupBox("Power Limit")
        limit_layout = QVBoxLayout(limit_group)
        limit_layout.setSpacing(8)

        self._limit_check = QCheckBox("Warn when output exceeds the limit")
        self._limit_check.setToolTip(
            "Flag the TX power above when it is higher than the maximum below"
        )
        self._limit_check.stateChanged.connect(self._on_limit_toggled)
        limit_layout.addWidget(self._limit_check)

        limit_form = _make_form()
        self._forms.append(limit_form)
        self._limit_spin = _WattSpinBox()
        self._limit_spin.setRange(0.001, 1500.0)
        self._limit_spin.setValue(5.0)
        self._limit_spin.setSuffix(" W")
        self._limit_spin.setDecimals(3)
        self._limit_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._limit_spin.setEnabled(False)
        self._limit_spin.setToolTip(_LIMIT_OFF_TIP)
        self._limit_spin.valueChanged.connect(self._on_limit_changed)
        self._max_power_label = QLabel("Max power:")
        self._max_power_label.setBuddy(self._limit_spin)
        limit_form.addRow(self._max_power_label, self._limit_spin)
        limit_layout.addLayout(limit_form)

        # QRP presets (each one also turns the limit on).
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(6)
        presets = (
            ("QRPp", QRP_LIMITS.qrpp_watts, "QRPp: {} or less"),
            ("CW", QRP_LIMITS.qrp_cw_watts, "QRP for CW and digital: {}"),
            ("SSB", QRP_LIMITS.qrp_ssb_watts, "QRP for phone: {}"),
        )
        self._preset_buttons = []
        for name, watts, tip in presets:
            btn = QPushButton(f"{name} {format_watts(watts)}")
            btn.setToolTip(f"Limit to {tip.format(format_watts(watts))}")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(
                lambda _checked=False, w=watts: self._set_quick_limit(w)
            )
            preset_layout.addWidget(btn)
            self._preset_buttons.append(btn)
        limit_layout.addLayout(preset_layout)

        layout.addWidget(limit_group)

        # --- Miles per watt -----------------------------------------------
        mpw_group = QGroupBox("Miles Per Watt")
        mpw_layout = QVBoxLayout(mpw_group)
        mpw_layout.setSpacing(8)
        mpw_form = _make_form()
        self._forms.append(mpw_form)

        self._distance_spin = QSpinBox()
        self._distance_spin.setRange(1, 20000)
        self._distance_spin.setValue(500)
        self._distance_spin.setSuffix(" mi")
        self._distance_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._distance_spin.setToolTip("Distance to the station you worked")
        self._distance_spin.valueChanged.connect(self._update_mpw_display)
        self._distance_label = QLabel("Distance:")
        self._distance_label.setBuddy(self._distance_spin)
        mpw_form.addRow(self._distance_label, self._distance_spin)

        self._mpw_power_spin = _WattSpinBox()
        self._mpw_power_spin.setRange(0.001, 100)
        self._mpw_power_spin.setDecimals(3)
        self._mpw_power_spin.setValue(5.0)
        self._mpw_power_spin.setSuffix(" W")
        self._mpw_power_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._mpw_power_spin.setToolTip("Transmit power used for the contact")
        self._mpw_power_spin.valueChanged.connect(self._update_mpw_display)
        self._mpw_power_label = QLabel("Power:")
        self._mpw_power_label.setBuddy(self._mpw_power_spin)
        mpw_form.addRow(self._mpw_power_label, self._mpw_power_spin)

        qso_row = QHBoxLayout()
        qso_row.setSpacing(8)
        self._mpw_label = QLabel("100 MPW")
        t.set_role(self._mpw_label, "value")
        self._mpw_label.setToolTip("Distance divided by power for this contact")
        qso_row.addWidget(self._mpw_label, 1)
        self._log_qso_btn = QPushButton("Log QSO")
        self._log_qso_btn.setToolTip("Add this contact to the miles-per-watt tally")
        self._log_qso_btn.clicked.connect(self._on_log_qso)
        qso_row.addWidget(self._log_qso_btn)
        self._this_qso_label = QLabel("This QSO:")
        mpw_form.addRow(self._this_qso_label, qso_row)
        mpw_layout.addLayout(mpw_form)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)
        best_caption = QLabel("BEST")
        t.set_role(best_caption, "caption")
        stats_row.addWidget(best_caption)
        self._best_mpw_label = QLabel("0 MPW")
        t.set_role(self._best_mpw_label, "value")
        stats_row.addWidget(self._best_mpw_label)
        stats_row.addSpacing(12)
        count_caption = QLabel("QSOS")
        t.set_role(count_caption, "caption")
        stats_row.addWidget(count_caption)
        self._qso_count_label = QLabel("0")
        t.set_role(self._qso_count_label, "value")
        stats_row.addWidget(self._qso_count_label)
        stats_row.addStretch(1)
        self._stats_row = QWidget()
        self._stats_row.setLayout(stats_row)
        stats_row.setContentsMargins(0, 0, 0, 0)
        mpw_layout.addWidget(self._stats_row)

        self._mpw_feedback = QLabel(
            "No QSOs logged yet. Enter a contact's distance and power, then "
            "press Log QSO."
        )
        self._mpw_feedback.setWordWrap(True)
        t.set_role(self._mpw_feedback, "hint")
        mpw_layout.addWidget(self._mpw_feedback)

        layout.addWidget(mpw_group)

        # Stretch at bottom
        layout.addStretch()

        self._align_form_labels()

        # Initial state: the calculator computed its output before we were
        # connected to it, so show that result now.
        self._update_mpw_display()
        self._refresh_power(self._amp_calc.get_output_dbm())

    def _align_form_labels(self) -> None:
        """Line up every group's fields on one left edge, and centre each
        form label on its field."""
        labels = [
            self._mode_label,
            self._max_power_label,
            self._distance_label,
            self._mpw_power_label,
            self._this_qso_label,
        ]
        for label in labels:
            label.ensurePolished()
        width = max(
            max(label.sizeHint().width() for label in labels),
            self._amp_calc.label_column_width(),
        )
        for label in labels:
            label.setMinimumWidth(width)
        self._amp_calc.set_label_column_width(width)
        for form in self._forms:
            _match_label_heights(form)

    # ------------------------------------------------------------------
    # Power / limit
    # ------------------------------------------------------------------

    def _refresh_power(self, dbm: float) -> None:
        """Show ``dbm`` in the TX power card and check it against the limit."""
        self._current_dbm = float(dbm)
        mode = self._mode_combo.currentText()
        self._power_display.set_power(dbm, mode)
        self._update_limit_status()

    def _update_limit_status(self) -> None:
        t = _theme()
        limit = self._qrp.get_power_limit()
        if not self._limit_check.isChecked() or limit is None:
            self._limit_status.setVisible(False)
            return
        watts = dbm_to_watts(self._current_dbm)
        if within_limit(watts, limit):
            text = f"✓ Within your {format_watts(limit)} limit"
            tone = "success"
        else:
            over_db = 10.0 * math.log10(max(watts, 1e-12) / max(limit, 1e-12))
            # A non-breaking space keeps "3.0 dB" together when wrapping.
            text = (
                f"\u26a0 Exceeds your {format_watts(limit)} limit "
                f"by {over_db:.1f}\u00a0dB"
            )
            tone = "danger"
        self._limit_status.setText(text)
        t.set_tone(self._limit_status, tone)
        self._limit_status.setVisible(True)

    def _on_mode_changed(self, mode: str):
        """Handle mode change."""
        self._refresh_power(self._current_dbm)

    def _on_limit_toggled(self, state: int):
        """Handle limit checkbox toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self._limit_spin.setEnabled(enabled)
        self._limit_spin.setToolTip(_LIMIT_TIP if enabled else _LIMIT_OFF_TIP)

        if enabled:
            self._qrp.set_power_limit(self._limit_spin.value())
            self.power_limit_changed.emit(self._limit_spin.value())
        else:
            self._qrp.disable_power_limit()
        self._update_limit_status()

    def _on_limit_changed(self, value: float):
        """Handle limit value change."""
        if self._limit_check.isChecked():
            self._qrp.set_power_limit(value)
            self.power_limit_changed.emit(value)
        self._update_limit_status()

    def _set_quick_limit(self, watts: float):
        """Set a quick power limit."""
        self._limit_spin.setValue(watts)
        self._limit_check.setChecked(True)
        self._qrp.set_power_limit(watts)
        self._update_limit_status()

    def _on_calc_power_changed(self, dbm: float):
        """Handle calculator power change."""
        self._refresh_power(dbm)

    # ------------------------------------------------------------------
    # Miles per watt
    # ------------------------------------------------------------------

    def _on_log_qso(self):
        """Log a QSO for MPW tracking."""
        distance = self._distance_spin.value()
        power = self._mpw_power_spin.value()

        previous_best = self._qrp.get_statistics()["best_mpw"]
        mpw = self._qrp.log_qso(distance, power)
        self._update_mpw_display()

        summary = f"Logged {distance} mi on {format_watts(power)} = {mpw:,.0f} MPW."
        if mpw > previous_best:
            self._mpw_feedback.setText(f"{summary} New best!")
            _theme().set_tone(self._mpw_feedback, "success")
        else:
            self._mpw_feedback.setText(summary)
            _theme().set_tone(self._mpw_feedback, None)

    def _update_mpw_display(self):
        """Update miles-per-watt display."""
        distance = self._distance_spin.value()
        power = self._mpw_power_spin.value()
        current_mpw = distance / power if power > 0 else 0

        self._mpw_label.setText(f"{current_mpw:,.0f} MPW")

        stats = self._qrp.get_statistics()
        self._best_mpw_label.setText(f"{stats['best_mpw']:,.0f} MPW")
        self._qso_count_label.setText(str(stats["total_qsos"]))
        self._stats_row.setVisible(stats["total_qsos"] > 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_controller(self) -> QRPController:
        """Get the QRP controller instance."""
        return self._qrp

    def set_power(self, dbm: float):
        """Set current power for display."""
        self._refresh_power(dbm)


__all__ = [
    "PowerDisplayWidget",
    "AmplifierCalculator",
    "QRPPanel",
    "classify_qrp",
    "format_dbm",
    "format_watts",
    "qrp_limit_for_mode",
    "within_limit",
]

"""
QRP (Low Power) Operations Panel.

Provides controls for QRP operation:
- The calculated station output in watts/mW/dBm, its QRP class, and the
  legal power limit and transmit privilege at the tuned frequency
- Amplifier chain calculator with presets, saturation and DC input
- A power limit (the transmitter refuses to go over it)
- Miles-per-watt tally (miles or kilometres) with the QSOs logged
- Contest exchange ("599 CA 5W") to copy
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QGuiApplication
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
        QLineEdit,
        QListWidget,
        QListWidgetItem,
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

from ..qrp import (
    QRP_LIMITS,
    AmplifierStage,
    QRPController,
    dbm_to_watts,
    get_qro_amplifier_chain,
    get_qrp_amplifier_chain,
)

# Modes held to the stricter CW/digital QRP limit (matches QRPController).
_CW_DIGITAL_MODES = (
    "CW",
    "RTTY",
    "PSK",
    "FT8",
    "FT4",
    "JT65",
    "WSPR",
    "DATA",
    "DIGITAL",
)
# Power within this many dB of a limit counts as "at" the limit. Whole-dB
# steps land just beside round watt values (+37 dBm is 5.01 W), which would
# otherwise show "5.0 W" next to "Above the 5 W QRP limit".
_LIMIT_TOLERANCE_DB = 0.05
# The US amateur limit where a segment names none (97.313(b)).
_AMATEUR_LIMIT_W = 1500.0
_KM_PER_MILE = 1.609344

_LIMIT_TIP = "Maximum transmit power you intend to use"
_LIMIT_OFF_TIP = (
    "Turn on “Warn when output exceeds the limit” (or pick a preset) "
    "to set a maximum power."
)
_AUTO_MODE = "Auto (follow demodulation)"
# Demodulation mode -> QRP mode, for Auto.
_DEMOD_TO_QRP = {"CW": "CW", "USB": "SSB", "LSB": "SSB", "AM": "AM", "FM": "FM"}
# QRP mode -> what the frequency manager calls it.
_QRP_TO_TX_MODE = {
    "CW": "CW",
    "SSB": "SSB",
    "FM": "FM",
    "AM": "AM",
    "FT8": "DATA",
    "RTTY": "DATA",
    "Digital": "DATA",
}
_MAX_QSOS = 200  # kept in the tally (the list shows the newest first)


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


def _finite(value: Any) -> float:
    """A saved number (ValueError for NaN or infinity, TypeError for what
    isn't a number)."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _spin_value(value: Any, spin: Any) -> int:
    """A saved number as a whole value inside ``spin``'s range (a QSpinBox
    raises OverflowError beyond 32 bits). ValueError for NaN, TypeError for
    what isn't a number."""
    number = float(value)
    if math.isnan(number):
        raise ValueError("not a number")
    return int(min(max(number, spin.minimum()), spin.maximum()))


# (smallest value in watts, watts per unit, unit). The thresholds sit just
# under 1 so a value that would round to "1000" of a unit uses the next unit
# up: 0.9996 W reads "1 W", not "1000 mW".
_POWER_UNITS = (
    (0.9995, 1.0, "W"),
    (0.9995e-3, 1e-3, "mW"),
    (0.9995e-6, 1e-6, "µW"),
    (0.9995e-9, 1e-9, "nW"),
)


def _unit_for(watts: float) -> Tuple[float, str]:
    scale, unit = 1e-12, "pW"
    for threshold, unit_scale, unit_name in _POWER_UNITS:
        if watts >= threshold:
            return unit_scale, unit_name
    return scale, unit


def format_watts(watts: float) -> str:
    """Power in the panel's one watt format.

    One decimal below 100 of a unit, none above, trailing ".0" dropped:
    ``"5 W"``, ``"1.3 W"``, ``"12.6 W"``, ``"250 mW"``, ``"1 mW"``.
    """
    watts = float(watts)
    if watts <= 0.0 or not math.isfinite(watts):
        return "0 W"
    scale, unit = _unit_for(watts)
    value = watts / scale
    number = f"{value:.0f}" if value >= 99.95 else _trim(f"{value:.1f}")
    return f"{number} {unit}"


def _format_watts_decimals(watts: float, decimals: int) -> Tuple[str, float]:
    """``(text, value shown in watts)`` with ``decimals`` decimals."""
    scale, unit = _unit_for(watts)
    number = _trim(f"{watts / scale:.{decimals}f}")
    return f"{number} {unit}", float(number) * scale


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


def legal_power_limit(
    freq_hz: float, tx_mode: str = ""
) -> Tuple[Optional[float], bool]:
    """``(watts, general)``: the legal power limit for sending ``tx_mode`` at
    ``freq_hz`` with the license class set. A segment's own limit, else in
    an amateur band the general amateur limit (1500 W PEP; ``general`` is
    then True), else None. Whether transmitting is allowed there at all is
    the legality check's to say (this assumes it is)."""
    from ...core import frequency_manager

    limit = frequency_manager.get_power_limit(float(freq_hz), tx_mode)
    if limit is not None:
        return float(limit), False
    if any(
        band.start_hz <= freq_hz <= band.end_hz
        for band in frequency_manager.AMATEUR_BAND_PRIVILEGES
    ):
        return _AMATEUR_LIMIT_W, True
    return None, False


def within_limit(watts: float, limit: float) -> bool:
    """True when ``watts`` is at or below ``limit`` (to within 0.05 dB)."""
    return watts <= limit * 10.0 ** (_LIMIT_TOLERANCE_DB / 10.0)


def _near_limit(watts: float, limit: float) -> bool:
    """Within the tolerance of ``limit``, on either side."""
    if watts <= 0 or limit <= 0:
        return False
    return abs(10.0 * math.log10(watts / limit)) <= _LIMIT_TOLERANCE_DB


def format_class_watts(watts: float, mode: str = "CW") -> str:
    """The power as the class badge sees it, so number and badge agree.

    A power within 0.05 dB of a class limit reads as the limit ("5 W" for
    +37 dBm, which counts as QRP); otherwise as many decimals as it takes
    for the number to sit on the same side of every class limit as the
    power itself ("1.02 W" for +30.1 dBm: above the 1 W QRPp limit, so
    QRP, not "1 W").
    """
    watts = float(watts)
    if watts <= 0.0 or not math.isfinite(watts):
        return "0 W"
    limits = sorted(
        {QRP_LIMITS.qrpp_watts, qrp_limit_for_mode(mode), QRP_LIMITS.low_power_watts}
    )
    for limit in limits:
        if _near_limit(watts, limit):
            return format_watts(limit)
    text = format_watts(watts)
    shown = _format_watts_decimals(watts, 0 if text.split()[0].isdigit() else 1)[1]
    if all((shown <= lim) == (watts <= lim) for lim in limits):
        return text
    for decimals in (1, 2, 3):
        text, shown = _format_watts_decimals(watts, decimals)
        if all((shown <= lim) == (watts <= lim) for lim in limits):
            return text
    return text


def classify_qrp(watts: float, mode: str = "CW") -> Tuple[str, str, str]:
    """Classify ``watts`` for ``mode``.

    Returns:
        ``(label, tone, explanation)``; ``tone`` is a theme tone name.
    """
    mode = (mode or "CW").upper()
    limit = qrp_limit_for_mode(mode)
    # (Auto's "Digital", for I/Q, reads as words; the other modes are names.)
    mode = "digital modes" if mode == "DIGITAL" else mode
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


def default_distance_unit() -> str:
    """ "mi" or "km" for the system locale (miles, the unit of the
    miles-per-watt awards, when the locale doesn't say)."""
    try:
        from PyQt6.QtCore import QLocale

        locale = QLocale.system()
        if locale.name() in ("C", "POSIX"):
            return "mi"
        metric = locale.measurementSystem() == QLocale.MeasurementSystem.MetricSystem
        return "km" if metric else "mi"
    except Exception:  # pragma: no cover - Qt missing
        return "mi"


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
            "Station output worked out by the Amplifier Chain below: a "
            "calculation, not a measurement of what is transmitted"
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
        self._dbm_label.setToolTip("Calculated output in dBm (0 dBm = 1 mW)")
        layout.addWidget(self._dbm_label, 1, 0, 1, 2)

        # Why the badge says what it says.
        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        t.set_role(self._detail_label, "hint")
        layout.addWidget(self._detail_label, 2, 0, 1, 2)
        layout.setColumnStretch(0, 1)

        self.set_power(0.0)

    def set_power(
        self, dbm: float, mode: str = "CW", legal_limit_w: Optional[float] = None
    ) -> None:
        """Show ``dbm`` for ``mode``; above ``legal_limit_w`` the badge is red."""
        watts = dbm_to_watts(dbm)
        label, tone, detail = classify_qrp(watts, mode)
        if legal_limit_w is not None and not within_limit(watts, legal_limit_w):
            tone = "danger"
            detail = (
                f"Above the {format_watts(legal_limit_w)} legal limit at this "
                f"frequency. {detail}."
            )

        self._watts_label.setText(format_class_watts(watts, mode))
        self._dbm_label.setText(format_dbm(dbm))
        self._status_label.setText(label)
        self._detail_label.setText(detail)
        self._status_label.setToolTip(detail)

        t = _theme()
        t.set_tone(self._status_label, tone)
        t.set_tone(self._watts_label, tone)


# Amplifier chain presets: (name, stages or None for the SDR alone).
def _chain_presets() -> List[Tuple[str, Optional[List[AmplifierStage]]]]:
    return [
        ("SDR only", None),
        ("HackRF → 5 W QRP kit", get_qrp_amplifier_chain()),
        ("HackRF → 100 W linear", get_qro_amplifier_chain()),
    ]


_CUSTOM = "Custom"
# Stage settings when nothing else is chosen: (gain dB, max out dBm, efficiency).
_DEFAULT_DRIVER = (20, 30, 0.5)
_DEFAULT_PA = (10, 50, 0.5)


class _Stage:
    """One amplifier stage's widgets and figures."""

    def __init__(
        self, name: str, check: Any, gain: Any, max_out: Any, out: Any, note: Any
    ):
        self.name = name
        self.check = check
        self.gain = gain
        self.max_out = max_out
        self.out = out
        self.note = note
        self.efficiency = 0.5


class AmplifierCalculator(QWidget if HAS_PYQT6 else object):
    """Amplifier chain power calculator: SDR output, a driver and a PA,
    each with a gain and the most it can put out (it saturates there), the
    output and the DC input the stages draw."""

    if HAS_PYQT6:
        power_changed = pyqtSignal(float)  # Output power in dBm
        chain_changed = pyqtSignal()  # any setting (to save it)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._presets = _chain_presets()
        self._applying = False
        self._user_sdr_dbm = 0.0
        self._sdr_locked = False
        # A connected HackRF's estimate as given (the field shows it to 0.1
        # dB; the figures use it exactly, as the transmitter's checks do).
        self._locked_dbm = 0.0
        self._sdr_source_tip = ""
        self._setup_ui()
        self.apply_preset(self._presets[0][0])

    def _setup_ui(self):
        t = _theme()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        layout.setColumnStretch(1, 1)

        preset_label = QLabel("Preset:")
        layout.addWidget(preset_label, 0, 0)
        self._preset_combo = QComboBox()
        for name, stages in self._presets:
            self._preset_combo.addItem(name)
            index = self._preset_combo.count() - 1
            self._preset_combo.setItemData(
                index, self._preset_tip(name, stages), Qt.ItemDataRole.ToolTipRole
            )
        self._preset_combo.addItem(_CUSTOM)
        self._preset_combo.setItemData(
            self._preset_combo.count() - 1,
            "Your own stages (chosen by itself when you change one)",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._preset_combo.setToolTip(
            "Common station setups; the stages below show what they hold"
        )
        self._preset_combo.currentTextChanged.connect(self._on_preset_chosen)
        preset_label.setBuddy(self._preset_combo)
        layout.addWidget(self._preset_combo, 0, 1, 1, 2)

        # Input power (SDR output)
        sdr_label = QLabel("SDR output:")
        self._label_column = [preset_label, sdr_label]
        layout.addWidget(sdr_label, 1, 0)
        self._input_spin = QDoubleSpinBox()
        self._input_spin.setRange(-90.0, 20.0)
        self._input_spin.setDecimals(1)
        self._input_spin.setSingleStep(1.0)
        self._input_spin.setValue(0)
        self._input_spin.setSuffix(" dBm")
        self._input_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._sdr_tip = (
            "RF output of the SDR (exciter). A HackRF One gives up to about "
            "+15 dBm depending on frequency, TX gain and its amplifier."
        )
        self._input_spin.setToolTip(self._sdr_tip)
        self._input_spin.valueChanged.connect(self._on_sdr_edited)
        sdr_label.setBuddy(self._input_spin)
        layout.addWidget(self._input_spin, 1, 1)
        self._input_watts = _readout("1 mW")
        layout.addWidget(self._input_watts, 1, 2)

        self._stages: List[_Stage] = []
        row = 2
        for name, accessible, (gain, max_out, _eff) in (
            ("Driver", "Driver gain", _DEFAULT_DRIVER),
            ("PA", "PA gain", _DEFAULT_PA),
        ):
            check = QCheckBox(f"{name}:")
            check.setToolTip(
                "Include a driver amplifier stage"
                if name == "Driver"
                else "Include the final power amplifier (PA)"
            )
            check.stateChanged.connect(self._on_stage_edited)
            layout.addWidget(check, row, 0)
            gain_spin = QSpinBox()
            gain_spin.setRange(0, 40)
            gain_spin.setValue(gain)
            gain_spin.setSuffix(" dB")
            gain_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            gain_spin.setToolTip(f"Gain of the {name.lower()} stage")
            # Labelled by a checkbox, which cannot be a buddy: name it directly.
            gain_spin.setAccessibleName(accessible)
            gain_spin.valueChanged.connect(self._on_stage_edited)
            layout.addWidget(gain_spin, row, 1)
            out = _readout("")
            layout.addWidget(out, row, 2)

            max_label = QLabel("max out:")
            max_label.setIndent(18)
            t.set_role(max_label, "muted")
            layout.addWidget(max_label, row + 1, 0)
            max_spin = QSpinBox()
            max_spin.setRange(0, 62)
            max_spin.setValue(max_out)
            max_spin.setSuffix(" dBm")
            max_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            max_spin.setToolTip(
                f"The most the {name.lower()} can put out (its saturated "
                "power): more drive doesn't raise its output beyond this"
            )
            max_spin.setAccessibleName(f"{name} maximum output")
            max_spin.valueChanged.connect(self._on_stage_edited)
            max_label.setBuddy(max_spin)
            layout.addWidget(max_spin, row + 1, 1)
            note = _readout("")
            layout.addWidget(note, row + 1, 2)
            self._label_column += [check, max_label]
            self._stages.append(_Stage(name, check, gain_spin, max_spin, out, note))
            row += 2

        driver, pa = self._stages
        self._driver_check, self._driver_spin, self._driver_out = (
            driver.check,
            driver.gain,
            driver.out,
        )
        self._pa_check, self._pa_spin, self._pa_out = pa.check, pa.gain, pa.out
        self._driver_max_spin, self._pa_max_spin = driver.max_out, pa.max_out

        # Result column must fit "(bypassed)", "saturated" and "79.4 mW".
        for label in (
            [self._input_watts]
            + [s.out for s in self._stages]
            + [s.note for s in self._stages]
        ):
            label.ensurePolished()
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance("(bypassed)"))

        # A little air between the stages and the totals.
        layout.setRowMinimumHeight(row, 2)
        row += 1

        # Output
        output_caption = QLabel("Output:")
        layout.addWidget(output_caption, row, 0)
        self._output_label = _readout("1 mW (0 dBm)", "value")
        self._output_label.setToolTip("Power at the antenna connector (calculated)")
        layout.addWidget(self._output_label, row, 1, 1, 2)
        row += 1

        # DC Power estimate
        dc_caption = QLabel("Est. DC input:")
        dc_caption.setToolTip(
            "Rough supply power the amplifier stages need, from their "
            "efficiency (50 % unless the preset says otherwise)"
        )
        layout.addWidget(dc_caption, row, 0)
        self._label_column += [output_caption, dc_caption]
        self._dc_label = _readout("0 W")
        self._dc_label.setToolTip(dc_caption.toolTip())
        layout.addWidget(self._dc_label, row, 1, 1, 2)

    @staticmethod
    def _preset_tip(name: str, stages: Optional[List[AmplifierStage]]) -> str:
        if not stages:
            return "The SDR's own output, no amplifier"
        parts = [
            f"{s.name}: {s.gain_db:g} dB, up to {format_watts(dbm_to_watts(s.max_output_dbm))}"
            for s in stages
        ]
        return f"{name}: " + "; ".join(parts)

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

    # ------------------------------------------------------------ presets

    def preset(self) -> str:
        return self._preset_combo.currentText()

    def apply_preset(self, name: str) -> None:
        """Set the stages to a preset ("SDR only", ... or "Custom": keep)."""
        stages = next((s for n, s in self._presets if n == name), "custom")
        self._applying = True
        try:
            index = self._preset_combo.findText(name)
            if index >= 0:
                self._preset_combo.setCurrentIndex(index)
            if stages != "custom":
                for widget, stage in zip(
                    self._stages, (stages or [None, None]), strict=False
                ):
                    if stage is None:
                        widget.check.setChecked(False)
                        continue
                    widget.check.setChecked(True)
                    widget.gain.setValue(int(round(stage.gain_db)))
                    widget.max_out.setValue(int(round(stage.max_output_dbm)))
                    widget.efficiency = float(stage.efficiency)
        finally:
            self._applying = False
        self._calculate()

    def _on_preset_chosen(self, name: str) -> None:
        if not self._applying:
            self.apply_preset(name)

    def _on_stage_edited(self, *_args) -> None:
        if self._applying:
            return
        if self.preset() != _CUSTOM:
            self._applying = True
            self._preset_combo.setCurrentIndex(self._preset_combo.findText(_CUSTOM))
            self._applying = False
        self._calculate()

    def _on_sdr_edited(self, value: float) -> None:
        if not self._sdr_locked and not self._applying:
            self._user_sdr_dbm = float(value)
        if not self._applying:
            self._calculate()

    # ------------------------------------------------------------ SDR input

    def set_sdr_output(self, dbm: Optional[float], tip: str = "") -> None:
        """The SDR output from a connected HackRF's TX power (read-only),
        or None to let the user set it again."""
        if dbm is None and not self._sdr_locked:
            return  # (already the user's own figure)
        if (
            dbm is not None
            and self._sdr_locked
            and abs(float(dbm) - self._locked_dbm) < 1e-6
            and tip == self._sdr_source_tip
        ):
            return  # unchanged
        self._sdr_source_tip = tip
        self._applying = True
        try:
            if dbm is None:
                self._sdr_locked = False
                self._input_spin.setReadOnly(False)
                self._input_spin.setButtonSymbols(
                    QDoubleSpinBox.ButtonSymbols.UpDownArrows
                )
                self._input_spin.setValue(self._user_sdr_dbm)
                self._input_spin.setToolTip(self._sdr_tip)
            else:
                self._sdr_locked = True
                self._locked_dbm = float(dbm)
                self._input_spin.setReadOnly(True)
                self._input_spin.setButtonSymbols(
                    QDoubleSpinBox.ButtonSymbols.NoButtons
                )
                self._input_spin.setValue(max(-90.0, min(20.0, float(dbm))))
                self._input_spin.setToolTip(
                    (tip or "From the connected HackRF's TX power")
                    + ". Change it in Ham Radio > Ham ID > TX Power."
                )
        finally:
            self._applying = False
        self._calculate(save=False)

    def sdr_locked(self) -> bool:
        return self._sdr_locked

    def sdr_dbm(self) -> float:
        """The SDR output the chain starts from: a connected HackRF's
        estimate (exactly), else the field's value."""
        if self._sdr_locked:
            return self._locked_dbm
        return float(self._input_spin.value())

    # ------------------------------------------------------------ figures

    def chain_output(
        self, input_dbm: Optional[float] = None
    ) -> Tuple[float, float, List[Tuple[float, bool]]]:
        """``(output dBm, DC input W, [(stage out dBm, saturated)])`` for an
        SDR output of ``input_dbm`` (:meth:`sdr_dbm` by default)."""
        current = float(self.sdr_dbm() if input_dbm is None else input_dbm)
        dc = 0.0
        stages = []
        for stage in self._stages:
            if not stage.check.isChecked():
                stages.append((current, False))
                continue
            wanted = current + stage.gain.value()
            ceiling = float(stage.max_out.value())
            saturated = wanted > ceiling + _LIMIT_TOLERANCE_DB
            current = min(wanted, ceiling)
            dc += dbm_to_watts(current) / max(stage.efficiency, 0.05)
            stages.append((current, saturated))
        return current, dc, stages

    def _calculate(self, save: bool = True):
        """Recalculate power chain (``save``: tell the panel to save it)."""
        t = _theme()
        input_dbm = self.sdr_dbm()
        self._input_watts.setText(format_watts(dbm_to_watts(input_dbm)))
        output_dbm, dc_power, figures = self.chain_output(input_dbm)
        for stage, (out_dbm, saturated) in zip(self._stages, figures, strict=True):
            on = stage.check.isChecked()
            stage.gain.setEnabled(on)
            stage.max_out.setEnabled(on)
            if on:
                stage.out.setText(format_watts(dbm_to_watts(out_dbm)))
                t.set_tone(stage.out, None)
                stage.note.setText("saturated" if saturated else "")
                t.set_tone(stage.note, "warning" if saturated else None)
                stage.note.setToolTip(
                    f"Driven past its {format_watts(dbm_to_watts(stage.max_out.value()))} "
                    "maximum: it can't put out more"
                    if saturated
                    else ""
                )
            else:
                stage.out.setText("(bypassed)")
                t.set_tone(stage.out, "muted")
                stage.note.setText("")
                stage.note.setToolTip("")

        self._output_label.setText(
            f"{format_watts(dbm_to_watts(output_dbm))} ({format_dbm(output_dbm)})"
        )
        self._dc_label.setText(format_watts(dc_power))
        self.power_changed.emit(output_dbm)
        if save and not self._applying:
            self.chain_changed.emit()

    def get_output_dbm(self) -> float:
        """Get calculated output power in dBm."""
        return self.chain_output()[0]

    # ------------------------------------------------------------ state

    def state(self) -> Dict[str, Any]:
        """The chain's settings (to save)."""
        return {
            "preset": self.preset(),
            "sdr_dbm": self._user_sdr_dbm,
            "stages": [
                [s.check.isChecked(), s.gain.value(), s.max_out.value(), s.efficiency]
                for s in self._stages
            ],
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore :meth:`state` (unknown or broken entries are skipped,
        numbers out of a control's range are clamped to it)."""
        self._applying = True
        try:
            try:
                self._user_sdr_dbm = _finite(state.get("sdr_dbm", 0.0))
            except (TypeError, ValueError):
                self._user_sdr_dbm = 0.0
            if not self._sdr_locked:
                self._input_spin.setValue(self._user_sdr_dbm)
            stages = state.get("stages")
            if isinstance(stages, list):
                for widget, saved in zip(self._stages, stages, strict=False):
                    try:
                        on, gain, max_out, eff = saved
                        gain_db = _spin_value(gain, widget.gain)
                        max_dbm = _spin_value(max_out, widget.max_out)
                        efficiency = min(max(_finite(eff), 0.05), 1.0)
                    except (TypeError, ValueError):
                        continue
                    widget.check.setChecked(bool(on))
                    widget.gain.setValue(gain_db)
                    widget.max_out.setValue(max_dbm)
                    widget.efficiency = efficiency
            preset = str(state.get("preset", ""))
            index = self._preset_combo.findText(preset)
            if index >= 0:
                self._preset_combo.setCurrentIndex(index)
        finally:
            self._applying = False
        self._calculate()


class QRPPanel(QWidget if HAS_PYQT6 else object):
    """
    Complete QRP operations panel.

    Provides:
    - Calculated output in multiple formats, its QRP class and the legal
      limit at the tuned frequency
    - Amplifier calculator with presets
    - TX power limit (the transmitter refuses to go over it)
    - Miles-per-watt tally with the QSOs logged
    - Contest exchange
    """

    if HAS_PYQT6:
        power_limit_changed = pyqtSignal(float)  # New limit in watts
        state_changed = pyqtSignal(dict)  # settings and tally (to save)
        exchange_copied = pyqtSignal(str)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._qrp = QRPController()
        self._current_dbm = 0.0
        self._restoring = False
        self._freq_hz: Optional[float] = None
        self._demod = "FM"
        self._legal_limit: Optional[float] = None
        self._qsos: List[Dict[str, Any]] = []
        self._undo_qsos: Optional[List[Dict[str, Any]]] = None
        self._rst_source: Optional[Callable[[], Optional[str]]] = None
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

        # --- Calculated output -------------------------------------------
        power_group = QGroupBox("Calculated Output")
        power_group.setToolTip(
            "What the station would put out, worked out from the Amplifier "
            "Chain below: a calculation, not a measurement"
        )
        power_layout = QVBoxLayout(power_group)
        power_layout.setSpacing(8)

        self._power_display = PowerDisplayWidget()
        power_layout.addWidget(self._power_display)

        # The license's privilege and power limit at the tuned frequency.
        self._legal_label = QLabel("")
        self._legal_label.setWordWrap(True)
        self._legal_label.setAccessibleName("Legal limit here")
        t.set_role(self._legal_label, "hint")
        self._legal_label.setVisible(False)
        power_layout.addWidget(self._legal_label)

        self._limit_status = QLabel("")
        self._limit_status.setWordWrap(True)
        t.set_role(self._limit_status, "callout", "success")
        self._limit_status.setVisible(False)
        power_layout.addWidget(self._limit_status)

        mode_form = _make_form()
        self._forms = [mode_form]
        # Shows "Auto · SSB" closed and the full items open, like the
        # S-meter's report format.
        from .signal_meter_widget import _ReportModeCombo

        self._mode_combo = _ReportModeCombo()
        self._mode_combo.addItem(_AUTO_MODE)
        self._mode_combo.addItems(["CW", "SSB", "FM", "FT8", "RTTY", "AM"])
        self._mode_combo.setItemData(
            0,
            "Follow Demodulation > Mode: CW is CW, USB/LSB SSB, AM and FM as "
            "they are, I/Q digital",
            Qt.ItemDataRole.ToolTipRole,
        )
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
        self._amp_calc.chain_changed.connect(self._emit_state)
        amp_layout.addWidget(self._amp_calc)

        layout.addWidget(amp_group)

        # --- Power limit --------------------------------------------------
        limit_group = QGroupBox("Power Limit")
        limit_layout = QVBoxLayout(limit_group)
        limit_layout.setSpacing(8)

        self._limit_check = QCheckBox("Warn when output exceeds the limit")
        self._limit_check.setToolTip(
            "Flag the calculated output when it is higher than the maximum "
            "below; the HackRF then refuses to transmit"
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
        self._mpw_group = mpw_group
        mpw_layout = QVBoxLayout(mpw_group)
        mpw_layout.setSpacing(8)
        mpw_form = _make_form()
        self._forms.append(mpw_form)

        self._unit_combo = QComboBox()
        self._unit_combo.addItem("Miles", "mi")
        self._unit_combo.addItem("Kilometres", "km")
        self._unit_combo.setToolTip(
            "Distances in miles (miles per watt, as the QRP awards count) or "
            "kilometres (km per watt)"
        )
        index = self._unit_combo.findData(default_distance_unit())
        self._unit_combo.setCurrentIndex(max(index, 0))
        self._update_mpw_title()  # (kilometres from the start in a metric locale)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        self._unit_label = QLabel("Units:")
        self._unit_label.setBuddy(self._unit_combo)
        mpw_form.addRow(self._unit_label, self._unit_combo)

        self._distance_spin = QSpinBox()
        self._distance_spin.setRange(1, 40000)
        self._distance_spin.setValue(500)
        self._distance_spin.setSuffix(f" {self._unit()}")
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
        self._log_qso_btn.setToolTip("Add this contact to the tally")
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

        # The QSOs logged, newest first, with Delete / Reset / Undo.
        self._qso_list = QListWidget()
        self._qso_list.setAccessibleName("Logged QSOs")
        self._qso_list.setToolTip(
            "Contacts in the tally, newest first. Select one and press Delete "
            "to take it out."
        )
        self._qso_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._qso_list.itemSelectionChanged.connect(self._update_qso_buttons)
        self._qso_list.installEventFilter(self)  # the Delete key deletes too
        self._qso_list.setMaximumHeight(110)
        mpw_layout.addWidget(self._qso_list)

        list_buttons = QHBoxLayout()
        list_buttons.setSpacing(6)
        self._delete_qso_btn = QPushButton("Delete")
        self._delete_qso_btn.setToolTip("Take the selected QSOs out of the tally")
        self._delete_qso_btn.clicked.connect(self._delete_selected_qsos)
        list_buttons.addWidget(self._delete_qso_btn)
        self._reset_btn = QPushButton("Reset Tally")
        self._reset_btn.setToolTip("Start the tally again (Undo brings it back)")
        self._reset_btn.clicked.connect(self._reset_tally)
        list_buttons.addWidget(self._reset_btn)
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setToolTip("Bring back what the last Delete or Reset removed")
        self._undo_btn.clicked.connect(self._undo_qso_change)
        list_buttons.addWidget(self._undo_btn)
        list_buttons.addStretch(1)
        self._qso_buttons = QWidget()
        self._qso_buttons.setLayout(list_buttons)
        list_buttons.setContentsMargins(0, 0, 0, 0)
        mpw_layout.addWidget(self._qso_buttons)

        self._mpw_feedback = QLabel(
            "No QSOs logged yet. Enter a contact's distance and power, then "
            "press Log QSO."
        )
        self._mpw_feedback.setWordWrap(True)
        t.set_role(self._mpw_feedback, "hint")
        mpw_layout.addWidget(self._mpw_feedback)

        layout.addWidget(mpw_group)

        # --- Contest exchange ---------------------------------------------
        exchange_group = QGroupBox("Contest Exchange")
        exchange_group.setToolTip(
            "The QRP contest exchange: the report you give, your location and "
            "your power"
        )
        exchange_form = _make_form(exchange_group)
        self._forms.append(exchange_form)
        self._state_edit = QLineEdit()
        self._state_edit.setPlaceholderText("e.g. CA")
        self._state_edit.setMaxLength(6)
        self._state_edit.setToolTip(
            "Your state, province or other location for the exchange (optional)"
        )
        self._state_edit.textChanged.connect(self._on_state_edited)
        self._state_label = QLabel("Location:")
        self._state_label.setBuddy(self._state_edit)
        exchange_form.addRow(self._state_label, self._state_edit)

        exchange_row = QHBoxLayout()
        exchange_row.setSpacing(8)
        self._exchange_label = QLabel("599 1mW")
        t.set_role(self._exchange_label, "lcd-small")
        self._exchange_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        exchange_row.addWidget(self._exchange_label, 1)
        self._copy_exchange_btn = QPushButton("Copy")
        self._copy_exchange_btn.setToolTip("Copy the exchange to the clipboard")
        self._copy_exchange_btn.clicked.connect(self.copy_exchange)
        exchange_row.addWidget(self._copy_exchange_btn)
        self._exchange_caption = QLabel("Exchange:")
        exchange_form.addRow(self._exchange_caption, exchange_row)
        layout.addWidget(exchange_group)

        # Stretch at bottom
        layout.addStretch()

        self._align_form_labels()

        # The exchange follows the S-meter's report while the panel shows.
        self._exchange_timer = QTimer(self)
        self._exchange_timer.setInterval(1000)
        self._exchange_timer.timeout.connect(self._update_exchange)

        # Initial state: the calculator computed its output before we were
        # connected to it, so show that result now.
        self._update_mode_face()
        self._update_mpw_display()
        self._update_qso_list()
        self._refresh_power(self._amp_calc.get_output_dbm())

    def _align_form_labels(self) -> None:
        """Line up every group's fields on one left edge, and centre each
        form label on its field."""
        labels = [
            self._mode_label,
            self._max_power_label,
            self._unit_label,
            self._distance_label,
            self._mpw_power_label,
            self._this_qso_label,
            self._state_label,
            self._exchange_caption,
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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_exchange()
        self._exchange_timer.start()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        """Delete (or Backspace) in the QSO list deletes the selected QSOs,
        like its Delete button."""
        if (
            obj is getattr(self, "_qso_list", None)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and self._qso_list.selectedItems()
        ):
            self._delete_selected_qsos()
            return True
        return super().eventFilter(obj, event)

    def hideEvent(self, event) -> None:
        self._exchange_timer.stop()
        super().hideEvent(event)

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def mode(self) -> str:
        """The QRP mode in effect ("CW", "SSB", "FM", "AM", "FT8", "RTTY" or
        "Digital"), Auto resolved from the demodulation mode."""
        text = self._mode_combo.currentText()
        if text != _AUTO_MODE:
            return text
        demod = (self._demod or "").upper()
        if demod.startswith("NONE") or demod == "I/Q":
            return "Digital"
        return _DEMOD_TO_QRP.get(demod, "FM")

    def _class_mode(self) -> str:
        """The mode the QRP class is judged in (Digital: the CW/data 5 W)."""
        return self.mode()

    def _tx_modes(self) -> Tuple[str, ...]:
        """What the frequency manager calls the mode in effect, for the
        legality check and the legal limit. SSB is checked as the sideband
        actually used (the 60 m channels allow USB only); a manual SSB
        choice while receiving in another mode tries USB, then LSB."""
        mode = self.mode()
        if mode != "SSB":
            return (_QRP_TO_TX_MODE.get(mode, mode),)
        demod = (self._demod or "").upper()
        return (demod,) if demod in ("USB", "LSB") else ("USB", "LSB")

    def set_demod_mode(self, mode: str) -> None:
        """Demodulation > Mode changed (Auto follows it)."""
        self._demod = str(mode or "")
        self._update_mode_face()
        self._refresh_power(self._current_dbm)

    def _update_mode_face(self) -> None:
        if self._mode_combo.currentText() == _AUTO_MODE:
            self._mode_combo.set_face_text(f"Auto · {self.mode()}")
            self._mode_combo.setToolTip(
                f"Auto: {self.mode()} (from Demodulation > Mode). QRP means 5 W "
                "or less for CW and digital modes, 10 W or less for phone."
            )
        else:
            self._mode_combo.set_face_text(None)
            self._mode_combo.setToolTip(
                "Operating mode. QRP means 5 W or less for CW and digital "
                "modes, 10 W or less for phone (SSB, FM, AM)."
            )

    # ------------------------------------------------------------------
    # Power / limit
    # ------------------------------------------------------------------

    def set_station(
        self, freq_hz: Optional[float], demod_mode: Optional[str] = None
    ) -> None:
        """The tuned frequency (and demodulation mode): the legal limit and
        privilege line follows them (and the license class)."""
        self._freq_hz = None if freq_hz is None else float(freq_hz)
        if demod_mode is not None:
            self._demod = str(demod_mode)
            self._update_mode_face()
        self._refresh_power(self._current_dbm)

    def refresh_license(self) -> None:
        """The license class changed."""
        self._refresh_power(self._current_dbm)

    def _update_legal(self) -> None:
        """The license's privilege and power limit at the tuned frequency."""
        t = _theme()
        self._legal_limit = None
        if self._freq_hz is None:
            self._legal_label.setVisible(False)
            return
        from ...core import frequency_manager
        from ...gui.transmitter import describe_refusal, license_name, quiet_checks

        freq = self._freq_hz
        where = f"{freq / 1e6:.3f} MHz"
        lic = license_name()
        head = f"At {where} ({lic}, {self.mode()})"
        # The first mode name permitted here (or, if none is, the first).
        with quiet_checks():
            checks = [
                (m, *frequency_manager.is_tx_allowed(freq, 0.0, m))
                for m in self._tx_modes()
            ]
        tx_mode, allowed, reason = next((c for c in checks if c[1]), checks[0])
        if not allowed:
            why = describe_refusal(reason, freq, tx_mode)
            text = f"Not permitted at {where} in {self.mode()}: {why}."
            tone: Optional[str] = "danger"
        else:
            found, is_general = legal_power_limit(freq, tx_mode)
            limit = _AMATEUR_LIMIT_W if found is None else found
            general = (
                " (the general amateur limit)" if found is None or is_general else ""
            )
            self._legal_limit = float(limit)
            watts = dbm_to_watts(self._current_dbm)
            if within_limit(watts, limit):
                text = f"{head}: legal limit {format_watts(limit)} PEP{general}."
                tone = "success"
            else:
                text = (
                    f"{head}: legal limit {format_watts(limit)} PEP{general}: "
                    f"the calculated {format_watts(watts)} is over it."
                )
                tone = "danger"
        self._legal_label.setText(text)
        t.set_tone(self._legal_label, tone)
        self._legal_label.setVisible(True)

    def _refresh_power(self, dbm: float) -> None:
        """Show ``dbm`` in the calculated output card and check it against
        the limits."""
        self._current_dbm = float(dbm)
        self._update_legal()
        self._power_display.set_power(dbm, self._class_mode(), self._legal_limit)
        self._update_limit_status()
        self._update_exchange()

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
                f"⚠ Exceeds your {format_watts(limit)} limit "
                f"by {over_db:.1f} dB: the HackRF won't transmit"
            )
            tone = "danger"
        self._limit_status.setText(text)
        t.set_tone(self._limit_status, tone)
        self._limit_status.setVisible(True)

    def _on_mode_changed(self, mode: str):
        """Handle mode change."""
        self._update_mode_face()
        self._refresh_power(self._current_dbm)
        self._emit_state()

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
        self._emit_state()

    def _on_limit_changed(self, value: float):
        """Handle limit value change."""
        if self._limit_check.isChecked():
            self._qrp.set_power_limit(value)
            self.power_limit_changed.emit(value)
        self._update_limit_status()
        self._emit_state()

    def _set_quick_limit(self, watts: float):
        """Set a quick power limit."""
        self._limit_spin.setValue(watts)
        self._limit_check.setChecked(True)
        self._qrp.set_power_limit(watts)
        self._update_limit_status()
        self._emit_state()

    def _on_calc_power_changed(self, dbm: float):
        """Handle calculator power change."""
        self._refresh_power(dbm)

    # ------------------------------------------------------------------
    # Miles per watt
    # ------------------------------------------------------------------

    def _unit(self) -> str:
        return str(self._unit_combo.currentData() or "mi")

    def _per_watt_name(self) -> str:
        return "MPW" if self._unit() == "mi" else "km/W"

    def _distance_in_unit(self, km: float) -> float:
        return km if self._unit() == "km" else km / _KM_PER_MILE

    def _update_mpw_title(self) -> None:
        self._mpw_group.setTitle(
            "Miles Per Watt" if self._unit() == "mi" else "Kilometres Per Watt"
        )

    def _on_unit_changed(self, _index: int) -> None:
        """Show distances in the other unit (the value typed converts)."""
        unit = self._unit()
        value = self._distance_spin.value()
        converted = value * _KM_PER_MILE if unit == "km" else value / _KM_PER_MILE
        self._distance_spin.blockSignals(True)
        self._distance_spin.setValue(max(1, int(round(converted))))
        self._distance_spin.setSuffix(f" {unit}")
        self._distance_spin.blockSignals(False)
        self._update_mpw_title()
        self._update_mpw_display()
        self._update_qso_list()
        self._emit_state()

    def _distance_km(self) -> float:
        value = float(self._distance_spin.value())
        return value if self._unit() == "km" else value * _KM_PER_MILE

    def _on_log_qso(self):
        """Log a QSO for MPW tracking."""
        distance = self._distance_spin.value()
        power = self._mpw_power_spin.value()
        km = self._distance_km()
        previous_best = self._best_per_watt()
        entry = {
            "km": round(km, 3),
            "watts": float(power),
            "when": time.strftime("%Y-%m-%d %H:%M"),
        }
        self._qsos.append(entry)
        del self._qsos[:-_MAX_QSOS]
        self._undo_qsos = None
        self._sync_controller()
        per_watt = self._distance_in_unit(km) / power if power > 0 else 0.0
        self._update_mpw_display()
        self._update_qso_list()

        summary = (
            f"Logged {distance} {self._unit()} on {format_watts(power)} = "
            f"{per_watt:,.0f} {self._per_watt_name()}."
        )
        if per_watt > previous_best:
            self._mpw_feedback.setText(f"{summary} New best!")
            _theme().set_tone(self._mpw_feedback, "success")
        else:
            self._mpw_feedback.setText(summary)
            _theme().set_tone(self._mpw_feedback, None)
        self._emit_state()

    def _best_per_watt(self) -> float:
        best = 0.0
        for qso in self._qsos:
            if qso["watts"] > 0:
                best = max(best, self._distance_in_unit(qso["km"]) / qso["watts"])
        return best

    def _sync_controller(self) -> None:
        """The QRP controller's statistics, from the QSO list (always miles)."""
        qrp = self._qrp
        qrp._total_qsos = len(self._qsos)
        qrp._total_miles = sum(q["km"] / _KM_PER_MILE for q in self._qsos)
        qrp._best_mpw = max(
            (
                (q["km"] / _KM_PER_MILE) / q["watts"]
                for q in self._qsos
                if q["watts"] > 0
            ),
            default=0.0,
        )

    def _update_mpw_display(self):
        """Update miles-per-watt display."""
        distance = self._distance_spin.value()
        power = self._mpw_power_spin.value()
        current = distance / power if power > 0 else 0
        name = self._per_watt_name()

        self._mpw_label.setText(f"{current:,.0f} {name}")
        self._best_mpw_label.setText(f"{self._best_per_watt():,.0f} {name}")
        self._qso_count_label.setText(str(len(self._qsos)))
        self._stats_row.setVisible(bool(self._qsos))

    def _update_qso_list(self) -> None:
        self._qso_list.clear()
        unit, name = self._unit(), self._per_watt_name()
        for index in range(len(self._qsos) - 1, -1, -1):
            qso = self._qsos[index]
            distance = self._distance_in_unit(qso["km"])
            per_watt = distance / qso["watts"] if qso["watts"] > 0 else 0.0
            text = (
                f"{distance:,.0f} {unit} · {format_watts(qso['watts'])} · "
                f"{per_watt:,.0f} {name}"
            )
            item = QListWidgetItem(text)
            if qso.get("when"):
                item.setToolTip(f"Logged {qso['when']}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            self._qso_list.addItem(item)
        self._qso_list.setVisible(bool(self._qsos))
        self._update_qso_buttons()

    def _update_qso_buttons(self) -> None:
        has = bool(self._qsos)
        self._delete_qso_btn.setEnabled(bool(self._qso_list.selectedItems()))
        self._reset_btn.setEnabled(has)
        self._undo_btn.setVisible(self._undo_qsos is not None)
        self._qso_buttons.setVisible(has or self._undo_qsos is not None)

    def _delete_selected_qsos(self) -> None:
        indexes = sorted(
            {
                int(item.data(Qt.ItemDataRole.UserRole))
                for item in self._qso_list.selectedItems()
            }
        )
        if not indexes:
            return
        self._undo_qsos = list(self._qsos)
        for index in reversed(indexes):
            del self._qsos[index]
        self._after_tally_edit(
            f"Deleted {len(indexes)} QSO{'s' if len(indexes) != 1 else ''}. "
            "Undo brings it back."
        )

    def _reset_tally(self) -> None:
        if not self._qsos:
            return
        count = len(self._qsos)
        self._undo_qsos = list(self._qsos)
        self._qsos = []
        self._after_tally_edit(
            f"Tally reset ({count} QSO{'s' if count != 1 else ''}). Undo brings "
            "it back."
        )

    def _undo_qso_change(self) -> None:
        if self._undo_qsos is None:
            return
        self._qsos = self._undo_qsos
        self._undo_qsos = None
        self._after_tally_edit("Restored.")

    def _after_tally_edit(self, message: str) -> None:
        self._sync_controller()
        self._update_mpw_display()
        self._update_qso_list()
        self._mpw_feedback.setText(message)
        _theme().set_tone(self._mpw_feedback, None)
        self._emit_state()

    # ------------------------------------------------------------------
    # Contest exchange
    # ------------------------------------------------------------------

    def set_rst_source(self, source: Optional[Callable[[], Optional[str]]]) -> None:
        """Where the report comes from (the S-meter: the RST of the signal
        received, or None without one; the contest's 599/59 then)."""
        self._rst_source = source
        self._update_exchange()

    def exchange_text(self) -> str:
        """The exchange: report, location (if set) and the calculated power,
        e.g. "599 CA 5W" or "59 250mW"."""
        rst = None
        if self._rst_source is not None:
            try:
                rst = self._rst_source()
            except Exception:  # pragma: no cover - defensive
                rst = None
        if not rst:
            rst = "599" if self.mode() in ("CW", "FT8", "RTTY", "Digital") else "59"
        watts = dbm_to_watts(self._current_dbm)
        state = self._state_edit.text().strip().upper()
        return self._qrp.format_contest_exchange(rst, watts, state)

    def _update_exchange(self) -> None:
        if not hasattr(self, "_exchange_label"):
            return
        text = self.exchange_text()
        self._exchange_label.setText(text)
        self._exchange_label.setToolTip(
            f"{text}: the report you give (the S-meter's when a signal is "
            "received, else the contest's usual one), your location and your "
            "calculated power"
        )

    def _on_state_edited(self, _text: str) -> None:
        self._update_exchange()
        self._emit_state()

    def copy_exchange(self) -> str:
        """Copy the exchange to the clipboard (and return it)."""
        text = self.exchange_text()
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._copy_exchange_btn.setToolTip(f"Copied: {text}")
        self.exchange_copied.emit(text)
        return text

    # ------------------------------------------------------------------
    # State (saved between sessions)
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Settings and the QSO tally, to save."""
        mode = self._mode_combo.currentText()
        return {
            "chain": self._amp_calc.state(),
            "mode": "auto" if mode == _AUTO_MODE else mode,
            "limit_on": self._limit_check.isChecked(),
            "limit_w": self._limit_spin.value(),
            "units": self._unit(),
            "qsos": list(self._qsos),
            "location": self._state_edit.text().strip(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore :meth:`get_state` (what can't be read is skipped)."""
        if not isinstance(state, dict):
            return
        self._restoring = True
        try:
            chain = state.get("chain")
            if isinstance(chain, dict):
                self._amp_calc.set_state(chain)
            mode = str(state.get("mode", "auto"))
            index = self._mode_combo.findText(_AUTO_MODE if mode == "auto" else mode)
            if index >= 0:
                self._mode_combo.setCurrentIndex(index)
            try:
                self._limit_spin.setValue(float(state.get("limit_w", 5.0)))
            except (TypeError, ValueError):
                pass
            self._limit_check.setChecked(bool(state.get("limit_on", False)))
            units = state.get("units")
            index = self._unit_combo.findData(units)
            if index >= 0 and index != self._unit_combo.currentIndex():
                self._unit_combo.setCurrentIndex(index)
            qsos = []
            saved_qsos = state.get("qsos")
            if not isinstance(saved_qsos, list):
                saved_qsos = []
            for qso in saved_qsos:
                if not isinstance(qso, dict):
                    continue
                try:
                    entry = {
                        "km": _finite(qso["km"]),
                        "watts": _finite(qso["watts"]),
                        "when": str(qso.get("when", "")),
                    }
                except (TypeError, ValueError, KeyError):
                    continue
                if entry["km"] > 0 and entry["watts"] > 0:
                    qsos.append(entry)
            self._qsos = qsos[-_MAX_QSOS:]
            self._undo_qsos = None
            self._state_edit.setText(str(state.get("location", ""))[:6])
            self._sync_controller()
            self._update_mpw_display()
            self._update_qso_list()
            self._refresh_power(self._amp_calc.get_output_dbm())
        finally:
            self._restoring = False

    def _emit_state(self, *_args) -> None:
        if not self._restoring:
            self.state_changed.emit(self.get_state())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_controller(self) -> QRPController:
        """Get the QRP controller instance."""
        return self._qrp

    def set_power(self, dbm: float):
        """Set current power for display."""
        self._refresh_power(dbm)

    def set_sdr_output(self, dbm: Optional[float], tip: str = "") -> None:
        """A connected HackRF's estimated output as the chain's SDR output
        (read-only), or None for the user's own figure."""
        self._amp_calc.set_sdr_output(dbm, tip)

    def calculated_output_watts(self, sdr_dbm: Optional[float] = None) -> float:
        """The station's output through the amplifier chain, for an SDR
        output of ``sdr_dbm`` (the chain's own figure by default)."""
        return dbm_to_watts(self._amp_calc.chain_output(sdr_dbm)[0])

    def power_limit_watts(self) -> Optional[float]:
        """The power limit when it is on (None: off)."""
        return self._qrp.get_power_limit() if self._limit_check.isChecked() else None

    @staticmethod
    def within(watts: float, limit: float) -> bool:
        """``watts`` at or below ``limit`` (to within 0.05 dB)."""
        return within_limit(watts, limit)

    @staticmethod
    def legal_limit_watts(freq_hz: float, tx_mode: str = "") -> Optional[float]:
        """The legal power limit at ``freq_hz`` for ``tx_mode`` (what the
        Calculated Output card checks: a segment's own limit, else the
        general amateur limit in an amateur band; None elsewhere)."""
        return legal_power_limit(freq_hz, tx_mode)[0]

    @staticmethod
    def watts_text(watts: float) -> str:
        return format_watts(watts)


__all__ = [
    "PowerDisplayWidget",
    "AmplifierCalculator",
    "QRPPanel",
    "classify_qrp",
    "default_distance_unit",
    "format_class_watts",
    "format_dbm",
    "format_watts",
    "legal_power_limit",
    "qrp_limit_for_mode",
    "within_limit",
]

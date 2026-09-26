"""
Device control panel widget.

Provides controls for:
- Frequency tuning (entry with unit selector and quick-tune steps)
- Receiver settings: RF gain / AGC and bandwidth
- Demodulation mode, FM deviation and squelch
- Frequency presets (with TX lockout indicators)
- I/Q recording
- License profile (transmit privileges)

The sections live in an internal scroll area, so the panel degrades
gracefully (scrolls instead of crushing its group boxes) when it is short.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, QSize, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QStyle,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..core import frequency_manager as _frequency_manager_module
from ..core.frequency_manager import (
    AMATEUR_BAND_PRIVILEGES,
    POWER_HEADROOM_FACTOR,
    TX_POWER_WARNING,
    LicenseClass,
    get_frequency_manager,
)
from ..utils.tooltips import get_short_tip
from .themes import set_role, set_tone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tuning range of the frequency entry: the union of every supported
# receiver (RTL-SDR from 500 kHz, HackRF One up to 6 GHz). The Bookmarks
# frequency field uses the same range, so a bookmark always stores exactly
# the frequency the tuner showed.
MIN_FREQUENCY_HZ = 1e3
MAX_FREQUENCY_HZ = 6e9

# One arrow-key / wheel step of the frequency entry (matches the plots'
# Left/Right tuning step). Ctrl+wheel and Page Up/Down step ten times this.
FREQ_STEP_HZ = 10e3

# Display unit -> (multiplier, decimals). Every unit resolves to 1 Hz.
_UNITS: Dict[str, Tuple[float, int]] = {
    "Hz": (1.0, 0),
    "kHz": (1e3, 3),
    "MHz": (1e6, 6),
    "GHz": (1e9, 9),
}

# Units whose integer part gets digit grouping ("145,800,000 Hz" instead of
# an unreadable "145800000").
_GROUPED_UNITS = ("Hz", "kHz")

QUICK_TUNE_STEPS = (-1e6, -100e3, -10e3, 10e3, 100e3, 1e6)

BANDWIDTH_OPTIONS = (
    "10 kHz",
    "25 kHz",
    "50 kHz",
    "100 kHz",
    "200 kHz",
    "500 kHz",
    "1 MHz",
    "2 MHz",
    "2.4 MHz",
)
DEFAULT_BANDWIDTH = "200 kHz"

# Item text (read by the main window), item tooltip key.
DEMOD_MODES: Tuple[Tuple[str, str], ...] = (
    ("None (I/Q)", ""),
    ("AM", "am"),
    ("FM", "fm"),
    ("USB", "ssb"),
    ("LSB", "ssb"),
    ("CW", ""),
)

FM_DEVIATIONS: Tuple[Tuple[str, str], ...] = (
    ("5 kHz", "Narrowband FM: amateur, marine, NOAA weather, PMR"),
    ("12.5 kHz", "Mid deviation"),
    ("25 kHz", "Wide deviation (e.g. weather-satellite APT)"),
    ("75 kHz", "Broadcast (wideband) FM"),
)
DEFAULT_FM_DEVIATION = "25 kHz"

# Named like File > Save Recording's filters, which preselect the choice.
RECORDING_FORMATS = (
    "Complex Float32 I/Q (.cf32)",
    "WAV, 16-bit I/Q (.wav)",
    "SigMF (.sigmf-data)",
)

# (license class, combo text, TX privileges summary)
LICENSE_CLASSES: Tuple[Tuple[LicenseClass, str, str], ...] = (
    (LicenseClass.NONE, "None (license-free)", "CB, MURS and FRS only"),
    (
        LicenseClass.TECHNICIAN,
        "Technician",
        "VHF/UHF, 10 m and limited HF CW",
    ),
    (LicenseClass.GENERAL, "General", "most HF bands plus VHF/UHF"),
    (LicenseClass.AMATEUR_EXTRA, "Amateur Extra", "full amateur privileges"),
)

# Preset mode -> Mode combo entry (presets use "WFM" and "RAW", which the
# Mode list calls FM with a 75 kHz deviation and "None (I/Q)").
_PRESET_DEMOD = {
    "FM": "FM",
    "WFM": "FM",
    "AM": "AM",
    "USB": "USB",
    "LSB": "LSB",
    "CW": "CW",
    "RAW": "None (I/Q)",
}
# How a preset's mode is shown in its details, in the Mode list's words.
_PRESET_MODE_LABEL = {"WFM": "FM (broadcast)", "RAW": "None (I/Q)"}

# Compact dummy-load reminder shown in the panel (full text in the tooltip).
TX_POWER_REMINDER = (
    "⚠ Before transmitting, check your output power with a 50 Ω dummy load "
    "and a wattmeter."
)

# Widest slider readout; both readouts share it so the sliders line up.
_VALUE_LABEL_WIDEST = "-120 dBFS"

_FORMAT_TIP = (
    "File format for the recorded I/Q samples; File > Save Recording offers "
    "it first. These are raw radio samples, not audio."
)

_APPLY_PRESET_TIP = "Tune to this preset and set its bandwidth and demodulation mode"

_STATUS_READY = "Ready"
_STATUS_RECORDING = "● Recording"
_STATUS_PAUSED = "❚❚ Paused"
_STATUS_ARMED = "◌ Armed – waiting for the receiver"
_ARMED_TIP = "Samples are captured once the receiver starts (Space)"
_PAUSE_TEXT = "❚❚ Pause"
_RESUME_TEXT = "▶ Resume"
_RECORD_TEXT = "● Record"
_STOP_TEXT = "⏹ Stop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hz(text: str) -> float:
    """Parse ``"2.4 MHz"`` / ``"25 kHz"`` / ``"500 Hz"`` into Hz."""
    value, _, unit = text.strip().partition(" ")
    scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(unit.strip(), 1.0)
    return float(value) * scale


def _format_mhz(freq_hz: float) -> str:
    """``146.52e6 -> "146.520 MHz"``, ``137.9125e6 -> "137.9125 MHz"``."""
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def _format_bandwidth(bw_hz: float) -> str:
    """``500 -> "500 Hz"``, ``2700 -> "2.7 kHz"``, ``2.4e6 -> "2.4 MHz"``."""
    bw_hz = float(bw_hz)
    if bw_hz >= 1e6:
        return f"{bw_hz / 1e6:.3f}".rstrip("0").rstrip(".") + " MHz"
    if bw_hz >= 1e3:
        return f"{bw_hz / 1e3:.2f}".rstrip("0").rstrip(".") + " kHz"
    return f"{bw_hz:.0f} Hz"


_LISTEN_OK = "Listen ✓"


def _power_limit_tip(legal: float, effective: Optional[float]) -> str:
    """Tooltip explaining the legal power limit and the app's TX guard."""
    pct = int(round(POWER_HEADROOM_FACTOR * 100))
    effective = effective or legal * POWER_HEADROOM_FACTOR
    return (
        f"Legal limit: {legal:g} W at your transmitter output. The app only "
        f"blocks power above {pct}% of it ({effective:g} W) to allow for "
        "cable loss; stay within the legal limit itself."
    )


def _in_amateur_band(freq_hz: float) -> bool:
    """Whether ``freq_hz`` lies in any amateur band (for any license class)."""
    return any(
        band.start_hz <= freq_hz <= band.end_hz for band in AMATEUR_BAND_PRIVILEGES
    )


@contextmanager
def _quiet_frequency_manager() -> Iterator[None]:
    """Silence the frequency manager's log while the panel only *inspects*.

    ``is_tx_allowed`` logs a warning for every blocked frequency, which is
    right for a real transmit attempt but turned merely browsing presets
    into a stream of warnings in the Error Log.
    """
    log = logging.getLogger(_frequency_manager_module.__name__)

    def _drop(_record: logging.LogRecord) -> bool:
        return False

    log.addFilter(_drop)
    try:
        yield
    finally:
        log.removeFilter(_drop)


class _WheelGuard(QObject if HAS_PYQT6 else object):
    """Send wheel events over *unfocused* inputs to the scroll area.

    Without this, scrolling the panel with the mouse wheel changes whatever
    combo box, slider or spin box happens to pass under the pointer (e.g.
    retuning or changing the gain by accident). Click an input first to
    adjust it with the wheel.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()  # ignored events propagate to the parent
            return True
        return False


# ---------------------------------------------------------------------------
# Frequency entry
# ---------------------------------------------------------------------------


class _FrequencySpinBox(QDoubleSpinBox if HAS_PYQT6 else object):
    """Spin box that keeps 1 Hz resolution but shows at least three
    decimals and drops the trailing zeros after them.

    In MHz that reads "100.000" or "137.9125", like the toolbar, the Bookmarks
    list and every other MHz readout, instead of an unreadable "100.000000".
    Units with three decimals or fewer (Hz, kHz) are shown unchanged.
    """

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt API)
        text = super().textFromValue(value)
        whole, point, frac = text.partition(self.locale().decimalPoint())
        if not point or len(frac) <= 3:
            return text
        return f"{whole}{point}{frac.rstrip('0').ljust(3, '0')}"


class FrequencyInput(QWidget if HAS_PYQT6 else object):
    """Frequency input widget with unit selection.

    Typing a value and pressing Enter (or leaving the field) tunes once,
    rather than retuning on every keystroke. Switching the unit keeps the
    tuned frequency and only changes how it is shown.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._frequency_hz = 100e6

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Frequency value (in the selected unit)
        self._freq_input = _FrequencySpinBox()
        self._freq_input.setKeyboardTracking(False)
        self._freq_input.setAccelerated(True)
        self._freq_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._freq_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._freq_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._freq_input.setMinimumWidth(96)
        self._freq_input.setToolTip(
            get_short_tip("center_frequency")
            + "\nType a value and press Enter. Arrow keys step 10 kHz "
            "(Page Up/Down: 100 kHz); click first to step with the wheel."
        )
        self._freq_input.setAccessibleName("Center frequency")
        layout.addWidget(self._freq_input, 1)

        # Unit selector
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(list(_UNITS))
        self._unit_combo.setCurrentText("MHz")
        self._unit_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._unit_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._unit_combo.setToolTip(
            "Unit for the frequency entry (changing it keeps the tuned frequency)"
        )
        self._unit_combo.setAccessibleName("Frequency unit")
        layout.addWidget(self._unit_combo)

        self._apply_unit()
        self._freq_input.valueChanged.connect(self._on_value_changed)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)

    def _get_multiplier(self) -> float:
        """Get current unit multiplier."""
        return _UNITS.get(self._unit_combo.currentText(), _UNITS["MHz"])[0]

    def _apply_unit(self) -> None:
        """Show the current frequency in the selected unit (no signals)."""
        unit = self._unit_combo.currentText()
        mult, decimals = _UNITS.get(unit, _UNITS["MHz"])
        spin = self._freq_input
        spin.blockSignals(True)
        spin.setGroupSeparatorShown(unit in _GROUPED_UNITS)
        spin.setDecimals(decimals)
        spin.setRange(MIN_FREQUENCY_HZ / mult, MAX_FREQUENCY_HZ / mult)
        spin.setSingleStep(FREQ_STEP_HZ / mult)
        spin.setValue(self._frequency_hz / mult)
        spin.blockSignals(False)

    def _on_value_changed(self, value: float):
        """Handle value change."""
        # Every unit resolves to 1 Hz; rounding drops float noise
        # (145.8 * 1e6 = 145800000.00000003).
        self._frequency_hz = float(round(value * self._get_multiplier()))
        self.frequency_changed.emit(self._frequency_hz)

    def _on_unit_changed(self, index: int):
        """Handle unit change: same frequency, new unit."""
        self._apply_unit()

    def set_frequency(self, freq_hz: float):
        """Set frequency in Hz (clamped to the tuning range, no signal).

        Values that are not finite numbers (NaN, infinity, e.g. from a
        corrupted setting or ``-f nan``) are ignored: the entry keeps the
        frequency it had.
        """
        try:
            freq_hz = float(freq_hz)
        except (TypeError, ValueError):
            return
        if not math.isfinite(freq_hz):
            return
        self._frequency_hz = min(max(freq_hz, MIN_FREQUENCY_HZ), MAX_FREQUENCY_HZ)
        mult = self._get_multiplier()
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(self._frequency_hz / mult)
        self._freq_input.blockSignals(False)

    def get_frequency(self) -> float:
        """Get frequency in Hz."""
        return self._frequency_hz


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------


class ControlPanel(QWidget if HAS_PYQT6 else object):
    """
    Device control panel.

    Provides controls for frequency, presets, gain, bandwidth, demodulation,
    squelch, recording and the transmit license profile.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)
        gain_changed = pyqtSignal(float)
        bandwidth_changed = pyqtSignal(float)
        demod_changed = pyqtSignal(str)
        recording_started = pyqtSignal(str)  # format
        recording_stopped = pyqtSignal()
        # Pause (True) / resume (False). The Pause button is only enabled
        # when something is connected to this signal.
        recording_paused = pyqtSignal(bool)
        license_changed = pyqtSignal(object)  # LicenseClass
        squelch_changed = pyqtSignal(float)  # dB
        agc_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._form_labels: List[QLabel] = []
        self._wheel_guard = _WheelGuard(self)
        # A recording is "armed" while it waits for the receiver to start.
        self._armed = False
        self._setup_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup UI elements."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Everything scrolls vertically; the panel never scrolls sideways.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        outer.addWidget(self._scroll)

        content = QWidget()
        self._content = content
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 6)
        layout.setSpacing(8)

        # Where to listen first: Presets sit right under the frequency they
        # set (the quickest way for a newcomer to find something to hear),
        # then how to listen (mode, squelch) and the receiver settings. The
        # occasional sections (recording, license) come last.
        layout.addWidget(self._build_frequency_group())
        layout.addWidget(self._build_presets_group())
        layout.addWidget(self._build_demod_group())
        layout.addWidget(self._build_receiver_group())
        layout.addWidget(self._build_recording_group())
        layout.addWidget(self._build_license_group())
        layout.addStretch()

        self._scroll.setWidget(content)

        self._align_form_labels()
        self._install_wheel_guard()

        # Initial state
        self._populate_preset_categories()
        self._sync_license_combo()
        self._update_fm_dev_enabled()
        self._update_record_controls(False)

    def minimumSizeHint(self) -> "QSize":  # noqa: N802 (Qt API)
        """Short (it scrolls) but never narrower than its content.

        The scroll area never scrolls sideways, so without this a splitter
        could squeeze the panel below its content width and clip the right
        edge of every group (the tool tabs alone allow ~290 px).
        """
        hint = super().minimumSizeHint()
        scroll = self._scroll
        bar = scroll.verticalScrollBar().sizeHint().width()
        if bar <= 0:
            bar = scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        width = self._content.minimumSizeHint().width() + bar + 2 * scroll.frameWidth()
        return QSize(max(hint.width(), width), hint.height())

    @staticmethod
    def _make_form(group: "QGroupBox") -> "QFormLayout":
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        return form

    def _add_row(self, form: "QFormLayout", text: str, field) -> "QLabel":
        """Add a labelled row, remembering the label for column alignment."""
        label = QLabel(text)
        if isinstance(field, QWidget):
            label.setBuddy(field)
        form.addRow(label, field)
        self._form_labels.append(label)
        return label

    @staticmethod
    def _shrinkable_combo(combo: "QComboBox", min_chars: int = 8) -> "QComboBox":
        """Let a combo box shrink below its longest item (no sideways scroll)."""
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(min_chars)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    @staticmethod
    def _value_label(text: str, widest: str) -> "QLabel":
        """Right-aligned readout next to a slider, wide enough not to jitter."""
        label = QLabel(text)
        set_role(label, "value")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.ensurePolished()
        label.setMinimumWidth(label.fontMetrics().horizontalAdvance(widest) + 4)
        return label

    @staticmethod
    def _hint(text: str = "") -> "QLabel":
        label = QLabel(text)
        label.setWordWrap(True)
        set_role(label, "hint")
        return label

    def _align_form_labels(self) -> None:
        """Give every form's label column the same width, so fields line up."""
        if not self._form_labels:
            return
        for label in self._form_labels:
            label.ensurePolished()
        width = max(label.sizeHint().width() for label in self._form_labels)
        for label in self._form_labels:
            label.setMinimumWidth(width)

    def _install_wheel_guard(self) -> None:
        for kind in (QComboBox, QAbstractSpinBox, QSlider):
            for widget in self._content.findChildren(kind):
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                widget.installEventFilter(self._wheel_guard)

    # -- Frequency -------------------------------------------------------

    def _build_frequency_group(self) -> "QGroupBox":
        group = QGroupBox("Frequency")
        box = QVBoxLayout(group)
        box.setSpacing(8)

        self._freq_input = FrequencyInput()
        self._freq_input.frequency_changed.connect(self.frequency_changed)
        box.addWidget(self._freq_input)

        # Quick-tune steps: down steps | up steps
        steps = QHBoxLayout()
        steps.setSpacing(4)
        self._quick_tune_buttons: List[QPushButton] = []
        for i, offset in enumerate(QUICK_TUNE_STEPS):
            if i == len(QUICK_TUNE_STEPS) // 2:
                steps.addSpacing(6)
            btn = QPushButton(self._format_offset(offset))
            set_role(btn, "compact")
            # An explicit minimum (text + padding) lets the six buttons share
            # a 360 px column; otherwise styles impose a ~80 px button minimum.
            btn.ensurePolished()
            btn.setMinimumWidth(btn.fontMetrics().horizontalAdvance(btn.text()) + 12)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            direction = "up" if offset > 0 else "down"
            spoken = f"Tune {direction} {_format_bandwidth(abs(offset))}"
            btn.setToolTip(spoken)
            # Screen readers announce this rather than "+10k".
            btn.setAccessibleName(spoken)
            btn.clicked.connect(lambda _checked=False, o=offset: self._quick_tune(o))
            steps.addWidget(btn)
            self._quick_tune_buttons.append(btn)
        box.addLayout(steps)
        return group

    # -- Presets ---------------------------------------------------------

    def _build_presets_group(self) -> "QGroupBox":
        group = QGroupBox("Presets")
        form = self._make_form(group)

        self._category_combo = self._shrinkable_combo(QComboBox())
        self._category_combo.setToolTip("Group of frequency presets to choose from")
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        self._add_row(form, "Category:", self._category_combo)

        # The preset combo gets the whole field width: sharing its row with
        # the Apply button truncated names such as "NOAA APT (NOAA-19)" at
        # 360 px, which made the three NOAA presets indistinguishable.
        self._preset_combo = self._shrinkable_combo(QComboBox())
        self._preset_combo.setToolTip("Frequency preset (details are shown below)")
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._add_row(form, "Preset:", self._preset_combo)

        # Preset details and its transmit status
        self._preset_info = self._hint()
        form.addRow(self._preset_info)
        self._preset_tx = self._hint()
        form.addRow(self._preset_tx)

        apply_row = QHBoxLayout()
        apply_row.setContentsMargins(0, 0, 0, 0)
        apply_row.addStretch(1)
        self._apply_preset_btn = QPushButton("Apply Preset")
        self._apply_preset_btn.setToolTip(_APPLY_PRESET_TIP)
        self._apply_preset_btn.clicked.connect(self._apply_preset)
        apply_row.addWidget(self._apply_preset_btn)
        form.addRow(apply_row)
        return group

    # -- Receiver --------------------------------------------------------

    def _build_receiver_group(self) -> "QGroupBox":
        group = QGroupBox("Receiver")
        form = self._make_form(group)

        gain_row = QWidget()
        row = QHBoxLayout(gain_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, 50)
        self._gain_slider.setValue(20)
        self._gain_slider.setPageStep(5)
        self._gain_slider.setToolTip(get_short_tip("gain"))
        self._gain_slider.setAccessibleName("RF gain")
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        row.addWidget(self._gain_slider, 1)
        self._gain_label = self._value_label("20 dB", _VALUE_LABEL_WIDEST)
        row.addWidget(self._gain_label)
        self._add_row(form, "RF gain:", gain_row).setBuddy(self._gain_slider)

        self._agc_check = QCheckBox("Automatic gain (AGC)")
        self._agc_check.setToolTip(
            "Let the tuner set its gain automatically. The RF gain slider is "
            "disabled while AGC is on."
        )
        self._agc_check.toggled.connect(self._on_agc_changed)
        form.addRow("", self._agc_check)

        self._bw_combo = self._shrinkable_combo(QComboBox(), 6)
        self._bw_combo.addItems(list(BANDWIDTH_OPTIONS))
        self._bw_combo.setCurrentText(DEFAULT_BANDWIDTH)
        self._bw_combo.setToolTip(get_short_tip("bandwidth"))
        self._bw_combo.currentTextChanged.connect(self._on_bandwidth_changed)
        self._add_row(form, "Bandwidth:", self._bw_combo)
        return group

    # -- Demodulation ----------------------------------------------------

    def _build_demod_group(self) -> "QGroupBox":
        group = QGroupBox("Demodulation")
        form = self._make_form(group)

        self._demod_combo = self._shrinkable_combo(QComboBox(), 6)
        for index, (text, tip_key) in enumerate(DEMOD_MODES):
            self._demod_combo.addItem(text)
            tip = get_short_tip(tip_key) if tip_key else ""
            if text == "None (I/Q)":
                tip = "No audio: show the spectrum and record raw I/Q only."
            elif text == "CW":
                tip = "Morse code (continuous wave), heard as a tone."
            elif text == "USB":
                tip = f"Upper sideband. {tip} Usual for HF voice above 10 MHz."
            elif text == "LSB":
                tip = f"Lower sideband. {tip} Usual for HF voice below 10 MHz."
            self._demod_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._demod_combo.setToolTip("How the tuned signal is turned into audio")
        self._demod_combo.currentTextChanged.connect(self._on_demod_changed)
        self._add_row(form, "Mode:", self._demod_combo)

        self._fm_dev_combo = self._shrinkable_combo(QComboBox(), 6)
        for index, (text, tip) in enumerate(FM_DEVIATIONS):
            self._fm_dev_combo.addItem(text)
            self._fm_dev_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._fm_dev_combo.setCurrentText(DEFAULT_FM_DEVIATION)
        self._add_row(form, "FM deviation:", self._fm_dev_combo)

        squelch_row = QWidget()
        row = QHBoxLayout(squelch_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._squelch_slider = QSlider(Qt.Orientation.Horizontal)
        self._squelch_slider.setRange(-120, 0)
        self._squelch_slider.setValue(-80)
        self._squelch_slider.setPageStep(5)
        self._squelch_slider.setToolTip(
            "Audio is muted while the signal peak is below this level (dBFS). "
            "Compare with the LEVEL readout in the toolbar."
        )
        self._squelch_slider.setAccessibleName("Squelch threshold")
        self._squelch_slider.valueChanged.connect(self._on_squelch_changed)
        row.addWidget(self._squelch_slider, 1)
        self._squelch_label = self._value_label("-80 dBFS", _VALUE_LABEL_WIDEST)
        row.addWidget(self._squelch_label)
        self._add_row(form, "Squelch:", squelch_row).setBuddy(self._squelch_slider)
        return group

    # -- Recording -------------------------------------------------------

    def _build_recording_group(self) -> "QGroupBox":
        group = QGroupBox("Recording")
        form = self._make_form(group)

        self._format_combo = self._shrinkable_combo(QComboBox())
        self._format_combo.addItems(list(RECORDING_FORMATS))
        self._format_combo.setToolTip(_FORMAT_TIP)
        self._add_row(form, "Format:", self._format_combo)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        # Neutral at rest, red ("danger") only while recording, exactly like
        # the toolbar's Record button: red always means "recording now".
        self._record_btn = QPushButton(_RECORD_TEXT)
        self._record_btn.setCheckable(True)
        self._record_btn.setToolTip(
            "Start or stop capturing I/Q samples. Save them afterwards with "
            "File > Save Recording."
        )
        self._record_btn.toggled.connect(self._on_record_toggled)
        buttons.addWidget(self._record_btn, 1)

        self._pause_btn = QPushButton(_PAUSE_TEXT)
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        buttons.addWidget(self._pause_btn, 1)
        form.addRow(buttons)

        status = QHBoxLayout()
        status.setSpacing(8)
        self._record_status = QLabel(_STATUS_READY)
        set_role(self._record_status, "hint")
        # Wraps rather than widening the panel for the longer armed text.
        self._record_status.setWordWrap(True)
        status.addWidget(self._record_status, 1)
        self._record_time = QLabel("00:00:00")
        set_role(self._record_time, "lcd-small")
        self._record_time.setToolTip("Elapsed recording time (hh:mm:ss)")
        status.addWidget(self._record_time)
        form.addRow(status)
        return group

    # -- License ---------------------------------------------------------

    def _build_license_group(self) -> "QGroupBox":
        group = QGroupBox("License Profile")
        form = self._make_form(group)

        self._license_combo = self._shrinkable_combo(QComboBox())
        for license_class, text, summary in LICENSE_CLASSES:
            self._license_combo.addItem(text, license_class)
            self._license_combo.setItemData(
                self._license_combo.count() - 1,
                f"Transmit privileges: {summary}",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._license_combo.setToolTip(
            "Your amateur license class. It decides where transmitting is "
            "allowed; receiving is never restricted."
        )
        self._license_combo.currentIndexChanged.connect(self._on_license_changed)
        self._add_row(form, "Class:", self._license_combo)

        self._license_info = self._hint()
        form.addRow(self._license_info)

        self._tx_warning = QLabel(TX_POWER_REMINDER)
        self._tx_warning.setWordWrap(True)
        set_role(self._tx_warning, "callout", "warning")
        self._tx_warning.setToolTip(TX_POWER_WARNING)
        form.addRow(self._tx_warning)
        return group

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------

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
        new = self._freq_input.get_frequency()  # clamped to the tuning range
        if new != current:
            self.frequency_changed.emit(new)

    def set_frequency(self, freq_hz: float):
        """Set frequency (no ``frequency_changed`` signal)."""
        self._freq_input.set_frequency(freq_hz)

    # ------------------------------------------------------------------
    # Gain / AGC / bandwidth
    # ------------------------------------------------------------------

    def _on_gain_changed(self, value: int):
        """Handle gain change."""
        if self._agc_check.isChecked():
            # The tuner owns the gain while AGC is on; the new value is
            # applied when AGC is switched off.
            return
        self._gain_label.setText(f"{value} dB")
        self.gain_changed.emit(float(value))

    def _on_agc_changed(self, enabled: bool):
        """Handle AGC toggle."""
        enabled = bool(enabled)
        self._gain_slider.setEnabled(not enabled)
        if enabled:
            self._gain_label.setText("Auto")
            self._gain_slider.setToolTip(
                "Gain is set automatically while AGC is on. Untick AGC to set "
                "it by hand."
            )
        else:
            self._gain_label.setText(f"{self._gain_slider.value()} dB")
            self._gain_slider.setToolTip(get_short_tip("gain"))
        self.agc_changed.emit(enabled)
        if not enabled:
            # Back to manual: re-apply the slider's gain. (Enabling AGC used
            # to emit gain_changed(-1), which set a fixed 0 dB manual gain
            # on the device and so switched AGC straight back off.)
            self.gain_changed.emit(float(self._gain_slider.value()))

    def is_agc_enabled(self) -> bool:
        """Whether AGC is currently enabled."""
        return bool(self._agc_check.isChecked())

    def set_agc_enabled(self, enabled: bool) -> None:
        """Toggle AGC."""
        self._agc_check.setChecked(bool(enabled))

    def set_gain(self, gain_db: float):
        """Set gain."""
        self._gain_slider.setValue(int(round(gain_db)))

    def _on_bandwidth_changed(self, text: str):
        """Handle bandwidth change."""
        if text:
            self.bandwidth_changed.emit(_parse_hz(text))

    # ------------------------------------------------------------------
    # Demodulation / squelch
    # ------------------------------------------------------------------

    def _on_demod_changed(self, text: str):
        """Handle demodulation mode change."""
        self._update_fm_dev_enabled()
        self.demod_changed.emit(text)

    def _update_fm_dev_enabled(self) -> None:
        is_fm = self._demod_combo.currentText() == "FM"
        self._fm_dev_combo.setEnabled(is_fm)
        if is_fm:
            self._fm_dev_combo.setToolTip(
                "Peak FM deviation: sets the audio level of the FM "
                "demodulator. 5 kHz for narrowband FM, 75 kHz for broadcast."
            )
        else:
            self._fm_dev_combo.setToolTip("Only used in FM mode")

    def get_demod_mode(self) -> str:
        """The selected demodulation mode (a :data:`DEMOD_MODES` name)."""
        return self._demod_combo.currentText()

    def get_fm_deviation_text(self) -> str:
        """The FM deviation choice as shown, e.g. ``"75 kHz"``."""
        return self._fm_dev_combo.currentText()

    def get_fm_deviation(self) -> float:
        """Return the selected FM deviation in Hz (default 25 kHz)."""
        text = self._fm_dev_combo.currentText()  # e.g. "25 kHz"
        try:
            return float(text.split()[0]) * 1e3
        except (ValueError, IndexError):
            return 25e3

    def _on_squelch_changed(self, value: int):
        """Handle squelch slider change."""
        self._squelch_label.setText(f"{value} dBFS")
        self.squelch_changed.emit(float(value))

    def get_squelch_db(self) -> float:
        """Return the current squelch threshold in dB."""
        return float(self._squelch_slider.value())

    def set_squelch_db(self, db: float) -> None:
        """Set the squelch threshold in dB."""
        self._squelch_slider.setValue(int(round(db)))

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _pause_supported(self) -> bool:
        """True when something listens to :attr:`recording_paused`."""
        try:
            return self.receivers(self.recording_paused) > 0
        except (TypeError, RuntimeError):  # pragma: no cover - defensive
            return False

    def _update_record_controls(self, recording: bool) -> None:
        """Reflect the recording state on the Recording group (no signals)."""
        self._record_btn.setText(_STOP_TEXT if recording else _RECORD_TEXT)
        role = "danger" if recording else ""
        if (self._record_btn.property("role") or "") != role:
            set_role(self._record_btn, role or None)
        if not recording:
            self._armed = False
        self._format_combo.setEnabled(not recording)
        self._format_combo.setToolTip(
            "Can't change the format while recording" if recording else _FORMAT_TIP
        )

        can_pause = recording and self._pause_supported()
        if not recording or not can_pause:
            self._pause_btn.blockSignals(True)
            self._pause_btn.setChecked(False)
            self._pause_btn.blockSignals(False)
            self._pause_btn.setText(_PAUSE_TEXT)
        self._pause_btn.setEnabled(can_pause)
        if not recording:
            self._pause_btn.setToolTip("Start a recording to pause it")
        elif not can_pause:
            self._pause_btn.setToolTip("Pausing isn't available for this recording")
        else:
            self._pause_btn.setToolTip("Pause or resume capturing samples")

        self._update_record_status()
        if not recording:
            self._record_time.setText("00:00:00")

    def _update_record_status(self) -> None:
        recording = self._record_btn.isChecked()
        paused = recording and self._pause_btn.isChecked()
        armed = recording and self._armed and not paused
        if paused:
            text, tone = _STATUS_PAUSED, "warning"
        elif armed:
            text, tone = _STATUS_ARMED, "warning"
        elif recording:
            text, tone = _STATUS_RECORDING, "danger"
        else:
            text, tone = _STATUS_READY, None
        self._record_status.setText(text)
        self._record_status.setToolTip(_ARMED_TIP if armed else "")
        set_tone(self._record_status, tone)
        # The timer only runs while samples are captured; idle or armed it is
        # muted (the LCD's default amber read as a warning).
        set_tone(self._record_time, "muted" if armed else tone or "muted")

    def _on_record_toggled(self, checked: bool):
        """Handle record button toggle."""
        self._update_record_controls(checked)
        if checked:
            self.recording_started.emit(self._format_combo.currentText())
        else:
            self.recording_stopped.emit()

    def _on_pause_toggled(self, paused: bool) -> None:
        self._pause_btn.setText(_RESUME_TEXT if paused else _PAUSE_TEXT)
        self._update_record_status()
        self.recording_paused.emit(bool(paused))

    def is_recording_paused(self) -> bool:
        """Whether the current recording is paused."""
        return self._record_btn.isChecked() and self._pause_btn.isChecked()

    def update_record_time(self, seconds: int):
        """Update recording time display."""
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        self._record_time.setText(f"{hours:02d}:{minutes:02d}:{secs:02d}")

    def set_recording_armed(self, armed: bool) -> None:
        """Show whether the current recording is armed: started, but waiting
        for the receiver to run before any samples are captured.

        Only matters while recording; stopping the recording clears it.
        """
        armed = bool(armed)
        if armed != self._armed:
            self._armed = armed
            self._update_record_status()

    def is_recording_armed(self) -> bool:
        """Whether the panel shows the recording as armed (not capturing)."""
        return self._record_btn.isChecked() and self._armed

    def set_recording_state(self, recording: bool) -> None:
        """Reflect recording state on the panel without re-emitting signals.

        Lets the main window keep this panel's Record button in sync when
        recording is toggled from elsewhere (e.g. the toolbar action).
        """
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(bool(recording))
        self._record_btn.blockSignals(False)
        self._update_record_controls(bool(recording))

    def get_recording_format(self) -> str:
        """Return the currently selected recording format label."""
        return self._format_combo.currentText()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _populate_preset_categories(self):
        """Populate the preset category dropdown."""
        fm = get_frequency_manager()
        categories = fm.get_preset_categories()
        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        self._category_combo.addItems(categories)
        self._category_combo.blockSignals(False)
        self._on_category_changed(self._category_combo.currentText())

    def _on_category_changed(self, category: str):
        """Handle category selection change."""
        presets = get_frequency_manager().get_rx_presets(category) if category else []
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for preset in presets:
            self._preset_combo.addItem(preset.name)
            self._preset_combo.setItemData(
                self._preset_combo.count() - 1,
                f"{_format_mhz(preset.frequency_hz)}: {preset.description}",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._preset_combo.blockSignals(False)
        self._on_preset_changed(self._preset_combo.currentText())

    def _on_preset_changed(self, preset_name: str):
        """Show the selected preset's details and whether TX is allowed."""
        fm = get_frequency_manager()
        preset = fm.get_preset_by_name(preset_name) if preset_name else None
        self._apply_preset_btn.setEnabled(preset is not None)
        self._apply_preset_btn.setToolTip(
            _APPLY_PRESET_TIP if preset is not None else "Choose a preset first"
        )
        if preset is None:
            self._preset_info.setText("Choose a category and a preset to see it here.")
            self._preset_tx.setText("")
            self._preset_tx.setToolTip("")
            set_tone(self._preset_tx, None)
            return

        mode = _PRESET_MODE_LABEL.get(preset.mode, preset.mode)
        self._preset_info.setText(
            f"{_format_mhz(preset.frequency_hz)} · "
            f"{_format_bandwidth(preset.bandwidth_hz)} · {mode}\n"
            f"{preset.description}"
        )

        with _quiet_frequency_manager():
            allowed, reason = fm.is_tx_allowed(
                preset.frequency_hz, preset.bandwidth_hz, preset.mode
            )
        reason = reason or ""
        # Receiving is never restricted, so every line starts by saying so:
        # the transmit status must not read as "this preset won't work".
        tip = reason
        if allowed:
            legal = fm.get_power_limit(preset.frequency_hz, preset.mode)
            effective = fm.get_effective_power_limit(preset.frequency_hz, preset.mode)
            text = f"{_LISTEN_OK} · Transmit ✓ with your license"
            if legal:
                text = f"{_LISTEN_OK} · Transmit ✓ (legal limit {legal:g} W)"
                tip = _power_limit_tip(legal, effective)
            tone = "success"
        elif reason.startswith("LICENSE:"):
            # Not covered by the selected license class: informational, since
            # listening works the same.
            text = self._license_block_text(
                reason[len("LICENSE:") :].strip(), preset.frequency_hz, preset.mode
            )
            tone = None
        else:
            # Hardware lockout (GPS, aviation, marine distress ...)
            detail = reason.replace("TX BLOCKED:", "").split("[")[0].strip()
            band = detail.split(" - ")[0].strip()
            text = (
                f"{_LISTEN_OK} · Transmit locked out: {band} is protected"
                if band
                else f"{_LISTEN_OK} · Transmit locked out: protected frequency"
            )
            tone = "warning"
        self._preset_tx.setText(text)
        self._preset_tx.setToolTip(tip)
        set_tone(self._preset_tx, tone)

    @staticmethod
    def _license_block_text(reason: str, freq_hz: float = 0.0, mode: str = "") -> str:
        """Short, plain-language version of a license restriction."""
        lowered = reason.lower()
        prefix = f"{_LISTEN_OK} · Transmit:"
        outside = f"{prefix} not allowed here (outside the amateur bands)"
        if "no amateur license" in lowered:
            # Only an amateur frequency becomes usable with a license; FM
            # broadcast, aviation or paging never do.
            if freq_hz and not _in_amateur_band(freq_hz):
                return outside
            return f"{prefix} needs an amateur license"
        if "higher license class" in lowered:
            return f"{prefix} needs a higher license class"
        if "not in any amateur" in lowered:
            return outside
        if lowered.startswith("mode"):
            return f"{prefix} {mode or 'this mode'} isn't allowed in this band segment"
        return f"{prefix} not allowed ({reason})" if reason else f"{prefix} not allowed"

    def _closest_bandwidth_index(self, bw_hz: float) -> int:
        """Index of the narrowest bandwidth option that fits ``bw_hz``."""
        values = [
            _parse_hz(self._bw_combo.itemText(i)) for i in range(self._bw_combo.count())
        ]
        if not values:
            return -1
        fitting = [i for i, v in enumerate(values) if v >= bw_hz]
        if fitting:
            return min(fitting, key=lambda i: values[i])
        return max(range(len(values)), key=lambda i: values[i])

    def _apply_preset(self):
        """Apply the selected preset: frequency, bandwidth and mode."""
        preset_name = self._preset_combo.currentText()
        if not preset_name:
            return

        fm = get_frequency_manager()
        preset = fm.get_preset_by_name(preset_name)
        if not preset:
            return

        # Set frequency
        self._freq_input.set_frequency(preset.frequency_hz)
        self.frequency_changed.emit(self._freq_input.get_frequency())

        # Bandwidth: the narrowest option that fits the preset (the old
        # substring match picked 25 kHz for 2.7 kHz SSB and missed 2 MHz).
        index = self._closest_bandwidth_index(preset.bandwidth_hz)
        if index >= 0:
            self._bw_combo.setCurrentIndex(index)

        # FM deviation that suits the channel width
        if preset.mode in ("FM", "WFM"):
            if preset.mode == "WFM" or preset.bandwidth_hz >= 150e3:
                self._fm_dev_combo.setCurrentText("75 kHz")
            elif preset.bandwidth_hz >= 40e3:
                self._fm_dev_combo.setCurrentText("25 kHz")
            else:
                self._fm_dev_combo.setCurrentText("5 kHz")

        # Set demodulation mode
        if preset.mode in _PRESET_DEMOD:
            idx = self._demod_combo.findText(_PRESET_DEMOD[preset.mode])
            if idx >= 0:
                self._demod_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # License profile
    # ------------------------------------------------------------------

    def _sync_license_combo(self) -> None:
        """Show the frequency manager's current class (no signal)."""
        current = get_frequency_manager().get_license_class()
        index = self._license_combo.findData(current)
        self._license_combo.blockSignals(True)
        self._license_combo.setCurrentIndex(max(index, 0))
        self._license_combo.blockSignals(False)
        self._update_license_info()

    def _selected_license(self) -> LicenseClass:
        data = self._license_combo.currentData()
        return data if isinstance(data, LicenseClass) else LicenseClass.NONE

    def _update_license_info(self) -> None:
        license_class = self._selected_license()
        summary = next((s for lc, _t, s in LICENSE_CLASSES if lc == license_class), "")
        self._license_info.setText(f"Transmit privileges: {summary}.")

    def _on_license_changed(self, index: int):
        """Handle license class change."""
        license_class = self._selected_license()

        # Update frequency manager
        get_frequency_manager().set_license_class(license_class)
        self._update_license_info()

        # Emit signal
        self.license_changed.emit(license_class)

        # Refresh preset TX status display
        self._on_preset_changed(self._preset_combo.currentText())

    def set_license_class(self, license_class: LicenseClass):
        """Set the license class (for external control)."""
        idx = max(self._license_combo.findData(license_class), 0)
        if idx != self._license_combo.currentIndex():
            self._license_combo.setCurrentIndex(idx)  # -> _on_license_changed
        elif get_frequency_manager().get_license_class() != self._selected_license():
            # The combo already shows it but the frequency manager was changed
            # elsewhere: bring it back in line with what the panel shows.
            self._on_license_changed(idx)

    def get_license_class(self) -> LicenseClass:
        """Get the current license class."""
        return get_frequency_manager().get_license_class()
